#include <iostream>

#include "sigmodindex.h"

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [input file]" << std::endl;
		exit(0);
	}

	SigmodIndex<int, int, int> index;

	index.load_points(argv[1]);
	return 0;
}
