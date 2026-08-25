
A handy CheatSheet for TMUX and Docker commands you’ll use every day.

## Some useful TMUX instructions

- `tmux ls`/`tmux list-session` – List all sessions 

- `tmux new -s <session-ID>/<session-name>` – Create a new session with the given name  
  
- `tmux kill-session -t <session-ID>/<session-name>` – Kill the specified session 
  
- `tmux attach -t <session-ID>/<session-name>` – Attach to the specified session  
  
- `tmux switch -t <session-ID>/<session-name>` – Switch to the specified session  


## Some useful Docker instructions

- `docker --version` – Show Docker version information  
  
- `docker info` – Display system‑wide information (containers, images, etc.)  
  
- `docker help` – Get help for any Docker command  

- `docker run` – Create and start a new container 
  
- `docker ps` – List currently running containers  
  
- `docker ps -a` – List all containers (including stopped ones)  
  
- `docker start <container-ID>/<container-name>` – Start a stopped container  
  
- `docker stop <container-ID>/<container-name>` – Stop a running container  
  
- `docker restart <container-ID>/<container-name>` – Restart a container  
  
- `docker rm <container-ID>/<container-name>` – Remove a **stopped** container

- `docker rm -f <container-ID>/<container-name>` – Force remove **running** containers

- `docker exec <container-ID>/<container-name> [command]` – Run a command inside a running container  
  
- `docker logs <container-ID>` – View the logs of a container  


To Be Continued…