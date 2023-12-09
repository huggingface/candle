clean: 
	rm -f search

run: search
	./search
search:
	nvcc -o search search_sorted.cu search_sorted_kernel.cu