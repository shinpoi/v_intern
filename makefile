init:
	gcc -o move_point.so -shared -fPIC move_point.c
	tar -xzvf ./data/dataset.tar.gz -C ./data/
	echo "init complete!"
	
claer:
	rm -i *.npy
