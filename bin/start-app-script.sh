#!/bin/bash


# gRPC server will spawn on 50051
PORT=50051

echo Current FINE_TUNING_SERVICE_IP: $FINE_TUNING_SERVICE_IP

{ # Try to start up the server
  echo "Starting up the gRPC server..."
  nohup python bin/start-grpc-server.py & 
} || {
  echo "gRPC server initialization script failed. Is there already a local server running in the pod?"
}

# Even though we've updated the project environment variables, we still need to update 
# the local environment variables. This is because the FINE_TUNING_SERVICE_IP/PORT 
# local env variables are still stale from the previous application dashboard environment,
# if this application is restarting.
export FINE_TUNING_SERVICE_IP=$CDSW_IP_ADDRESS
export FINE_TUNING_SERVICE_PORT=$PORT

echo New FINE_TUNING_SERVICE_IP: $FINE_TUNING_SERVICE_IP

echo "Waiting 5 seconds..."
sleep 5


# Start up the streamlit application
echo "Starting up streamlit application..."
streamlit run main.py --server.port $1 --server.address 127.0.0.1

