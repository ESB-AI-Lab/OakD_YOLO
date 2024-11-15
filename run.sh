#!/usr/bin/env bash
# Define the image name
IMAGE_NAME="ghcr.io/kevin12j/oakd_yolo:latest"

# Check if the image exists locally
if ! sudo docker images -q "$IMAGE_NAME" > /dev/null; then
    echo "The image '$IMAGE_NAME' does not exist locally. Pulling the latest version..."
    sudo docker pull "$IMAGE_NAME"
else
    # Pull the latest image from the repository
    sudo docker pull "$IMAGE_NAME"

    # Get the image ID of the local version
    LOCAL_IMAGE_ID=$(sudo docker images -q "$IMAGE_NAME")

    # Get the image ID of the latest version
    LATEST_IMAGE_ID=$(sudo docker images --quiet --filter=reference="$IMAGE_NAME")

    # Compare the image IDs
    if [ "$LOCAL_IMAGE_ID" == "$LATEST_IMAGE_ID" ]; then
        echo "The local image '$IMAGE_NAME' is the most recent version."
    else
        echo "A newer version of the image '$IMAGE_NAME' is available. Updating..."
        
        # Pull the latest image again to ensure it's updated
        sudo docker pull "$IMAGE_NAME"
        
        if [ $? -eq 0 ]; then
            echo "Successfully updated the image '$IMAGE_NAME' to the latest version."
        else
            echo "Failed to update the image '$IMAGE_NAME'."
        fi
    fi
fi

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	echo "### DISPLAY environmental variable is already set: \"$DISPLAY\""
	# give docker root user X11 permissions
	xhost +si:localuser:root || sudo xhost +si:localuser:root
	
	# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
	XAUTH=/tmp/.docker.xauth
	xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
	chmod 777 $XAUTH

	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# check if sudo is needed
if [ $(id -u) -eq 0 ] || id -nG "$USER" | grep -qw "docker"; then
	SUDO=""
else
	SUDO="sudo"
fi

# Initialize an empty array for filtered arguments
filtered_args=()

# Loop through all provided arguments
for arg in "$@"; do
    if [[ "$arg" != "--csi2webcam" && "$arg" != --csi-capture-res=* && "$arg" != --csi-output-res=* ]]; then
        filtered_args+=("$arg")  # Add to the new array if not the argument to remove
    fi
    
    if [[ "$arg" = "--name" || "$arg" = --name* ]]; then
        HAS_CONTAINER_NAME=1
    fi
done

filtered_args+=("$IMAGE_NAME")

if [ -z "$HAS_CONTAINER_NAME" ]; then
    # Generate a unique container name
    BUILD_DATE_TIME=$(date +%Y%m%d_%H%M%S)
    CONTAINER_NAME="oak_yolo_container_${BUILD_DATE_TIME}"
    CONTAINER_NAME_FLAGS="--name ${CONTAINER_NAME}"
fi

# run the container
ARCH=$(uname -i)

if [ $ARCH = "aarch64" ]; then

	# this file shows what Jetson board is running
	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/nv_jetson_model

	( set -x ;

	$SUDO docker run --runtime nvidia -it --rm --network host \
		--shm-size=8g \
		--volume /tmp/argus_socket:/tmp/argus_socket \
		--volume /etc/enctune.conf:/etc/enctune.conf \
		--volume /etc/nv_tegra_release:/etc/nv_tegra_release \
		--volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
		--volume /var/run/dbus:/var/run/dbus \
		--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
		--volume /var/run/docker.sock:/var/run/docker.sock \
		--volume $ROOT/data:/data \
		-v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
		--device /dev/snd \
		--device /dev/bus/usb \
		--privileged -v /dev/bus/usb:/dev/bus/usb \
	        --device-cgroup-rule='c 189:* rmw' \
		$DISPLAY_DEVICE \
		$CONTAINER_NAME_FLAGS \
		"${filtered_args[@]}"
	)
else
	echo "Script Made For aarch64"
fi
