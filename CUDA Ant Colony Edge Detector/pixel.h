struct pixel{
	float tao;
	float ni;
	float val;
	bool ant;
	pixel *neigh[8];
	int neighCount;
	int antCount;
};
