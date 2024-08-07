#!/bin/bash


# gRPC server will spawn on 50051
PORT=50051

{ # Try to start up the server
  echo "Starting up the gRPC server..."
  nohup python bin/start-grpc-server.py & 
} || {
  echo "gRPC server initialization script failed. Is there already a local server running in the pod?"
}


echo "Waiting 5 seconds..."
sleep 5


# Start up the streamlit application
echo "Starting up streamlit application..."
streamlit run main.py --server.port $1 --server.address 127.0.0.1

