#!/usr/bin
export http_proxy=""

wget http://sendbag.9rum.cc/files/jason.yoon-1565134272-4924029-6d6c2d3130306b2e7a6970/ml-100k.zip
wget http://sendbag.9rum.cc/files/jason.yoon-1565134425-198702078-6d6c2d32306d2e7a6970/ml-20m.zip
wget http://sendbag.9rum.cc/files/jason.yoon-1565135529-33251838-74657874382e7a6970/text8.zip

export http_proxy=http://proxy.daumkakao.io:3128

unzip ml-100k.zip
unzip ml-20m.zip
unzip text8.zip

rm *.zip
