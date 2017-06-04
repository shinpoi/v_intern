all:
	gcc -o move_point.so -shared -fPIC move_point.c

claer_data:
	rm -i *.npy
