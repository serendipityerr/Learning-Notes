
A handy CheatSheet for TMUX and Docker commands you’ll use every day.

## Some useful TMUX instructions

- List all sessions  
    ```bash
    tmux ls
    tmux list-session
    ```

- Create a new session with the given name
    ```bash
    tmux new -s <session-name>
    ```

- Kill the specified session 
    ```bash
    tmux kill-session -t <session-ID>/<session-name>
    ```

- Attach to the specified session
    ```bash
    tmux attach -t <session-ID>/<session-name>
    ```

- Switch to the specified session
    ```bash
    tmux switch -t <session-ID>/<session-name>
    ```

## Some useful Docker instructions

- Show Docker version information
    ```bash
    docker --version
    ```

- Display system‑wide information (containers, images, etc.)  
    ```bash
    docker info
    ```

- Get help for any Docker command
    ```bash
    docker help
    ```

- Create and start a new container from the specified image
    ```bash
    docker run [image]
    ```
    
- List currently running containers
    ```bash
    docker ps
    ```

- List all containers (including stopped ones)
    ```bash
    docker ps -a
    ```
    
- Start a stopped container
    ```bash
    docker start <container-ID>/<container-name>
    ```
    
- Stop a running container
    ```bash
    docker stop <container-ID>/<container-name>
    ```
    
- Restart a container
    ```bash
    docker restart <container-ID>/<container-name>
    ```
    
- Remove a **stopped** container
    ```bash
    docker rm <container-ID>/<container-name>
    ```
    
- Force remove **running** containers
    ```bash
    docker rm -f <container-ID>/<container-name>
    ```
    
- Run a command inside a running container
    ```bash
    docker exec <container-ID>/<container-name> [command]
    ```
    
- View the logs of a container
    ```bash
    docker logs <container-ID>/<container-name>
    ```


To Be Continued…