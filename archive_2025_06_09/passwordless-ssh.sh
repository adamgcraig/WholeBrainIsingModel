#!/bin/sh
#
# passwordless	SSH	setup	among	all	GPU	nodes
#
> ~/.ssh/authorized_keys
> ~/.ssh/known_hosts
yes y | ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa >/dev/null
cp -p ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
for NODE in $(cat /etc/hosts |egrep hkbugpusrv[0-9]{2}$ |awk '{	print $2 }'); do
ssh -o "StrictHostKeyChecking no" $NODE hostname >/dev/null 2>&1
done