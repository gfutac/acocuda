struct position{
	int i, j;
	float pheromone;
	position *neigh[8];
	int neighCount;
	int antCount;
};
