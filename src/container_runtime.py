# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Container Runtime Abstraction

Provides a unified interface for Docker and Podman container runtimes,
allowing the benchmark to work with either container engine.
"""

import subprocess
from typing import List, Optional
from src.config_manager import config


class ContainerRuntime:
    """
    Abstraction layer for container runtime commands.

    Supports both Docker and Podman, automatically adjusting commands
    based on the configured runtime.
    """

    def __init__(self, runtime: Optional[str] = None):
        """
        Initialize container runtime.

        Args:
            runtime: Container runtime to use ('docker' or 'podman').
                    If None, reads from CONTAINER_RUNTIME config.
        """
        self.runtime = (runtime or config.get("CONTAINER_RUNTIME", "docker")).lower()

        if self.runtime not in ["docker", "podman"]:
            raise ValueError(f"Unsupported container runtime: {self.runtime}. Must be 'docker' or 'podman'")

        # Determine compose command
        if self.runtime == "docker":
            self.compose_cmd = "docker-compose"
            self.container_cmd = "docker"
        else:  # podman
            self.compose_cmd = "podman-compose"
            self.container_cmd = "podman"

    def get_compose_command(self) -> str:
        """Get the compose command (docker-compose or podman-compose)."""
        return self.compose_cmd

    def get_container_command(self) -> str:
        """Get the container command (docker or podman)."""
        return self.container_cmd

    def build_compose_run_cmd(self, docker_file: str, project_name: str,
                              service: str, user_id: str, group_id: str,
                              entrypoint: Optional[str] = None,
                              cmd_args: str = "") -> str:
        """
        Build a docker-compose/podman-compose run command.

        Args:
            docker_file: Path to docker-compose.yml file
            project_name: Project name for the compose command
            service: Service name to run
            user_id: User ID to run as
            group_id: Group ID to run as
            entrypoint: Optional entrypoint override (e.g., 'bash')
            cmd_args: Additional command arguments

        Returns:
            Complete compose run command string
        """
        base_cmd = f"{self.compose_cmd} -f {docker_file} -p {project_name} run --rm"

        # Podman handles user mapping differently - it uses --userns=keep-id by default
        # For compatibility, we still pass the user, but Podman may handle it differently
        user_flags = f"--user {user_id}:{group_id}"

        # Add HOME environment variable
        home_flag = "-e HOME=/code/rundir"

        # Build the command
        if entrypoint:
            return f"{base_cmd} {user_flags} {home_flag} --entrypoint {entrypoint} {cmd_args} {service}"
        else:
            return f"{base_cmd} {user_flags} {home_flag} {cmd_args} {service}"

    def build_compose_kill_cmd(self, docker_file: str, project_name: str, service: str) -> str:
        """
        Build a docker-compose/podman-compose kill command.

        Args:
            docker_file: Path to docker-compose.yml file
            project_name: Project name for the compose command
            service: Service name to kill

        Returns:
            Complete compose kill command string
        """
        return f"{self.compose_cmd} -f {docker_file} -p {project_name} kill {service}"

    def build_images_cmd(self, image_name: str) -> List[str]:
        """
        Build a command to check if an image exists.

        Args:
            image_name: Name of the image to check

        Returns:
            Command as list of strings
        """
        return [self.container_cmd, "images", "-q", image_name]

    def build_image_build_cmd(self, tag: str, dockerfile: str, context: str = ".") -> List[str]:
        """
        Build a command to build a container image.

        Args:
            tag: Tag for the built image
            dockerfile: Path to Dockerfile
            context: Build context path

        Returns:
            Command as list of strings
        """
        return [self.container_cmd, "build", "-t", tag, "-f", dockerfile, context]

    def build_volume_rm_cmd(self, volume_name: str) -> List[str]:
        """
        Build a command to remove a volume.

        Args:
            volume_name: Name of the volume to remove

        Returns:
            Command as list of strings
        """
        return [self.container_cmd, "volume", "rm", "-f", volume_name]

    def check_availability(self) -> bool:
        """
        Check if the configured container runtime is available.

        Returns:
            True if the runtime is available, False otherwise
        """
        try:
            # Check if the main container command exists
            subprocess.run(
                [self.container_cmd, "--version"],
                capture_output=True,
                check=True
            )

            # Check if compose command exists
            subprocess.run(
                [self.compose_cmd, "--version"],
                capture_output=True,
                check=True
            )

            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_runtime_name(self) -> str:
        """Get the name of the configured runtime."""
        return self.runtime

    def is_podman(self) -> bool:
        """Check if using Podman runtime."""
        return self.runtime == "podman"

    def is_docker(self) -> bool:
        """Check if using Docker runtime."""
        return self.runtime == "docker"


# Global container runtime instance
container_runtime = ContainerRuntime()
