#!/bin/bash
sed -i 's/\r$//' /root/sam_job.sh
nohup bash /root/sam_job.sh > /root/sam-job.log 2>&1 &
echo $! > /root/sam-job.pid
echo started
