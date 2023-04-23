# get id of container with the corresponding name and store to variable id
id=$(sudo docker ps -a | grep stablelm | awk '{print $1}')
# connect bash to container with id $id
sudo docker exec -it $id /bin/bash
