use cudarc::nccl::safe::Id;
use std::convert::TryInto;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

pub async fn run_nccl_id_server(port: u16, nccl_id: Id, num_workers: usize) -> std::io::Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    println!("NCCL ID Server listening on 0.0.0.0:{}", port);

    let nccl_id_bytes: &[i8; 128] = nccl_id.internal();
    let nccl_id_bytes = Arc::new(*nccl_id_bytes);

    let mut connected_workers = 0;
    while connected_workers < num_workers {
        match listener.accept() {
            Ok((mut stream, addr)) => {
                println!("Worker connected from: {}", addr);
                let nccl_id_bytes = Arc::clone(&nccl_id_bytes);

                let bytes_to_send: Vec<u8> = nccl_id_bytes.iter().map(|&x| x as u8).collect();
                if let Err(e) = stream.write_all(&bytes_to_send) {
                    eprintln!("Error sending NCCL ID to worker {}: {:?}", addr, e);
                } else {
                    connected_workers += 1;
                    println!(
                        "NCCL ID sent to worker {}. {}/{} workers connected.",
                        addr, connected_workers, num_workers
                    );
                }
            }
            Err(e) => {
                eprintln!("Error accepting connection: {:?}", e);
            }
        }
    }

    println!("NCCL ID sent to all {} workers", num_workers);
    Ok(())
}

pub async fn get_nccl_id_from_server(addr: SocketAddr) -> std::io::Result<Id> {
    let mut stream = TcpStream::connect(addr)?;
    let mut buffer = [0u8; 128];
    stream.read_exact(&mut buffer)?;

    let internal: [i8; 128] = buffer
        .iter()
        .map(|&b| b as i8)
        .collect::<Vec<i8>>()
        .try_into()
        .unwrap();
    Ok(Id::uninit(internal))
}
