use std::collections::{HashMap, VecDeque};
use std::os::fd::{AsFd, OwnedFd};
use std::rc::Rc;
use std::{mem, slice};

use anyhow::{bail, Result};
use nix::sys::epoll::EpollFlags;
use nix::sys::stat::{fstat, SFlag};
use nix::sys::uio::pread;

use crate::guest::bridge::common;
use crate::guest::bridge::common::{
    Client, CrossDomainResource, GemHandleFinalizer, MappedBlob, MessageResourceFinalizer,
    ProtocolHandler, StreamRecvResult, StreamSendResult,
};

const CROSS_DOMAIN_CHANNEL_TYPE_WAYLAND: u32 = 0x0001;
const CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB: u32 = 1;

/// Globals whose messages carry fds the cross-domain channel cannot translate
/// (pipes, DRM syncobjs, sync files, client-sent keymaps / gamma ramps) or whose
/// shm buffers the host writes. They are renamed in wl_registry.global, so
/// clients do not bind them.
const HIDDEN_GLOBALS: &[&str] = &[
    "wl_data_device_manager",
    "zwp_primary_selection_device_manager_v1",
    "gtk_primary_selection_device_manager",
    "zwlr_data_control_manager_v1",
    "ext_data_control_manager_v1",
    "wp_linux_drm_syncobj_manager_v1",
    "zwp_linux_explicit_synchronization_v1",
    "wp_security_context_manager_v1",
    "wl_drm",
    "zwlr_gamma_control_manager_v1",
    "zwp_virtual_keyboard_manager_v1",
    "zwp_input_method_manager_v2",
    "zwlr_export_dmabuf_manager_v1",
    "zwlr_screencopy_manager_v1",
    "ext_image_copy_capture_manager_v1",
];

/// Address space reserved for a pool on the host, so that wl_shm_pool.resize
/// (which the host applies to the same fd) stays inside the shadow memfd.
const POOL_MIN_CAPACITY: usize = 16 << 20;

/// A client wl_shm pool: the client's fd and the host memfd the compositor maps.
/// Buffer contents are copied client -> shadow when the buffer is attached.
struct Pool {
    client_fd: OwnedFd,
    shadow: MappedBlob,
}

enum Object {
    Registry,
    Compositor,
    Shm,
    Pool(Rc<Pool>),
    Buffer {
        pool: Rc<Pool>,
        offset: usize,
        len: usize,
    },
    Surface,
}

pub struct WaylandProtocolHandler {
    objects: HashMap<u32, Object>,
    /// Shadows of shm fds already sent, in fd order; taken by wl_shm.create_pool.
    pending_pools: VecDeque<Rc<Pool>>,
    send_resources: Vec<CrossDomainResource>,
    send_finalizers: Vec<WaylandResourceFinalizer>,
    recv_fds: Vec<OwnedFd>,
}

pub struct WaylandResourceFinalizer(GemHandleFinalizer);

impl MessageResourceFinalizer for WaylandResourceFinalizer {
    type Handler = WaylandProtocolHandler;

    fn finalize(self, client: &mut Client<Self::Handler>) -> Result<()> {
        self.0.finalize(client)
    }
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_ne_bytes(data[at..at + 4].try_into().unwrap())
}

/// Wayland string argument at `at`: (contents without NUL, padded wire length).
fn read_string(data: &[u8], at: usize) -> Result<(&[u8], usize)> {
    let len = read_u32(data, at) as usize;
    let padded = 4 + len.next_multiple_of(4);
    if len == 0 || at + padded > data.len() {
        bail!("malformed wayland string");
    }
    Ok((&data[at + 4..at + 4 + len - 1], padded))
}

impl WaylandProtocolHandler {
    /// Translate the fds received from the client in stream order: shm files
    /// get a host shadow, everything else must be a virtio-gpu dmabuf.
    fn take_client_fds(this: &mut Client<Self>) -> Result<()> {
        for fd in mem::take(&mut this.request_fds) {
            let stat = fstat(fd.as_fd())?;
            if SFlag::from_bits_truncate(stat.st_mode) & SFlag::S_IFMT == SFlag::S_IFREG {
                let capacity = (stat.st_size as usize * 4).max(POOL_MIN_CAPACITY);
                let shadow = this.alloc_host_shm(capacity)?;
                this.protocol_handler.send_resources.push(CrossDomainResource {
                    identifier: shadow.res_id,
                    identifier_type: CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB,
                    identifier_size: 0,
                });
                this.protocol_handler.pending_pools.push_back(Rc::new(Pool {
                    client_fd: fd,
                    shadow,
                }));
            } else {
                let (res, fin) = this.vgpu_id_from_prime(fd)?;
                this.protocol_handler.send_resources.push(res);
                this.protocol_handler
                    .send_finalizers
                    .push(WaylandResourceFinalizer(fin));
            }
        }
        Ok(())
    }

    fn process_request(&mut self, id: u32, opcode: u16, msg: &[u8]) -> Result<()> {
        let arg = |i: usize| read_u32(msg, 8 + 4 * i);
        let new = match (self.objects.get(&id), opcode) {
            // wl_display.get_registry
            (None, 1) if id == 1 => Some((arg(0), Object::Registry)),
            // wl_registry.bind
            (Some(Object::Registry), 0) => {
                let (interface, len) = read_string(msg, 12)?;
                let new_id = read_u32(msg, 12 + len + 4);
                match interface {
                    b"wl_compositor" => Some((new_id, Object::Compositor)),
                    b"wl_shm" => Some((new_id, Object::Shm)),
                    _ => None,
                }
            },
            // wl_compositor.create_surface
            (Some(Object::Compositor), 0) => Some((arg(0), Object::Surface)),
            // wl_shm.create_pool
            (Some(Object::Shm), 0) => {
                let Some(pool) = self.pending_pools.pop_front() else {
                    bail!("wl_shm.create_pool without a shm fd");
                };
                Some((arg(0), Object::Pool(pool)))
            },
            // wl_shm_pool.create_buffer
            (Some(Object::Pool(pool)), 0) => {
                let offset = arg(1) as usize;
                let len = arg(3) as usize * arg(4) as usize;
                if offset + len > pool.shadow.len {
                    bail!("wl_shm buffer beyond the shadow capacity");
                }
                Some((
                    arg(0),
                    Object::Buffer {
                        pool: pool.clone(),
                        offset,
                        len,
                    },
                ))
            },
            // wl_shm_pool.resize
            (Some(Object::Pool(pool)), 2) => {
                if arg(0) as usize > pool.shadow.len {
                    bail!("wl_shm_pool.resize beyond the shadow capacity");
                }
                None
            },
            // wl_surface.attach
            (Some(Object::Surface), 1) => {
                if let Some(Object::Buffer { pool, offset, len }) = self.objects.get(&arg(0)) {
                    // SAFETY: create_buffer checked `offset + len` against the mapping
                    let dst = unsafe { slice::from_raw_parts_mut(pool.shadow.ptr.add(*offset), *len) };
                    let mut done = 0;
                    while done < *len {
                        match pread(pool.client_fd.as_fd(), &mut dst[done..], (offset + done) as i64)? {
                            0 => break,
                            n => done += n,
                        }
                    }
                }
                None
            },
            _ => None,
        };
        if let Some((new_id, object)) = new {
            self.objects.insert(new_id, object);
        }
        Ok(())
    }
}

/// Requests that the bridge reads; all others are forwarded unparsed.
fn inspects_request(objects: &HashMap<u32, Object>, id: u32) -> bool {
    id == 1 || objects.contains_key(&id)
}

impl ProtocolHandler for WaylandProtocolHandler {
    type ResourceFinalizer = WaylandResourceFinalizer;

    const CHANNEL_TYPE: u32 = CROSS_DOMAIN_CHANNEL_TYPE_WAYLAND;

    fn new() -> Self {
        WaylandProtocolHandler {
            objects: HashMap::new(),
            pending_pools: VecDeque::new(),
            send_resources: Vec::new(),
            send_finalizers: Vec::new(),
            recv_fds: Vec::new(),
        }
    }

    fn process_recv_stream(
        this: &mut Client<Self>,
        data: &mut [u8],
        resources: &mut VecDeque<CrossDomainResource>,
    ) -> Result<StreamRecvResult> {
        for rsc in resources.drain(..) {
            if rsc.identifier_type != CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB {
                bail!("unsupported host identifier type {}", rsc.identifier_type);
            }
            let fd = this.virtgpu_id_to_memfd(rsc)?;
            this.protocol_handler.recv_fds.push(fd);
        }
        if data.len() < 8 {
            return Ok(StreamRecvResult::WantMore);
        }
        let id = read_u32(data, 0);
        let opcode = read_u32(data, 4) as u16;
        let size = (read_u32(data, 4) >> 16) as usize;
        if size < 8 {
            bail!("malformed wayland event");
        }
        let handler = &mut this.protocol_handler;
        let registry = matches!(handler.objects.get(&id), Some(Object::Registry));
        if (id == 1 && opcode == 1) || (registry && opcode == 0) {
            if data.len() < size {
                return Ok(StreamRecvResult::WantMore);
            }
            if id == 1 {
                // wl_display.delete_id
                handler.objects.remove(&read_u32(data, 8));
            } else {
                // wl_registry.global
                let (interface, _) = read_string(data, 12)?;
                if HIDDEN_GLOBALS.iter().any(|g| g.as_bytes() == interface) {
                    data[16] = b'_';
                }
            }
        }
        Ok(StreamRecvResult::Processed {
            consumed_bytes: size,
            fds: mem::take(&mut this.protocol_handler.recv_fds),
        })
    }

    fn process_send_stream(
        this: &mut Client<Self>,
        data: &mut [u8],
    ) -> Result<StreamSendResult<WaylandResourceFinalizer>> {
        Self::take_client_fds(this)?;
        if data.len() < 8 {
            return Ok(StreamSendResult::WantMore);
        }
        let id = read_u32(data, 0);
        let opcode = read_u32(data, 4) as u16;
        let size = (read_u32(data, 4) >> 16) as usize;
        if size < 8 {
            bail!("malformed wayland request");
        }
        if inspects_request(&this.protocol_handler.objects, id) {
            if data.len() < size {
                return Ok(StreamSendResult::WantMore);
            }
            this.protocol_handler
                .process_request(id, opcode, &data[..size])?;
        }
        Ok(StreamSendResult::Processed {
            consumed_bytes: size,
            resources: mem::take(&mut this.protocol_handler.send_resources),
            finalizers: mem::take(&mut this.protocol_handler.send_finalizers),
        })
    }

    fn process_vgpu_extra(_this: &mut Client<Self>, cmd: u8) -> Result<()> {
        bail!("unexpected cross-domain command {cmd}")
    }

    fn process_fd_extra(_this: &mut Client<Self>, _fd: u64, _events: EpollFlags) -> Result<()> {
        Ok(())
    }
}

pub fn start_wlbridge(sock_path: &str) {
    common::bridge_loop::<WaylandProtocolHandler>(sock_path)
}
