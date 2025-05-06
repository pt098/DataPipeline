#!/bin/bash

# Script to generate SSH keys for MPI container communication

# Create keys directory if it doesn't exist
mkdir -p ../keys

# Generate SSH key pair if it doesn't exist
if [ ! -f "../keys/id_rsa" ]; then
    echo "Generating SSH keys for container communication..."
    ssh-keygen -t rsa -b 4096 -f ../keys/id_rsa -N ""
    chmod 600 ../keys/id_rsa
    chmod 644 ../keys/id_rsa.pub
    echo "SSH keys generated successfully."
else
    echo "SSH keys already exist."
fi

echo "Keys are stored in the 'keys' directory." 