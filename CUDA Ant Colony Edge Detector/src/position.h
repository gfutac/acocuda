struct position{
	float pheromone;
	position *neigh[8];
	int neighCount;
	int antCount;
	int i, j;
};
