#define GLM_FORCE_CUDA
#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm.hpp>

#define f_separation 1.0f
#define f_alignment 1.0f
#define f_cohesion 1.0f
#define range 10.0f
#define velocity 2.0f
//#define N 1 << 12

struct Boid {
	glm::vec3 position{};
	glm::vec3 heading{};
};

__constant__ constexpr int N = 1 << 8;

///////////////////////////////
///		COPYING MEMORY IS TAKING FOREVER. DO MORE ON DEVICE EVEN IF IT'S SINGLE THREADED. BREAK OUT INTO MORE KERNELS, SORRY
////////////////////////////////


//__global__
//void update_distance(std::size_t i, const Boid* boids, float* distances)
//{
//	int index = threadIdx.x;
//	int stride = blockDim.x;
//
//	for (std::size_t j{ index }; j < N; j += stride)
//	{
//		//if (j == i) { continue; }
//
//		distances[j] = glm::distance(boids[i].position, boids[j].position);
//		//distances[j] = 68;
//	}
//}

//__global__
//void update_aggregates(std::size_t i, const Boid* boids, Boid* new_boids, float* distances, float* debug)
//{
//	//const float f_separation = 1;
//	//const float f_alignment = 1;
//	//const float f_cohesion = 1;
//	//const float range = 5;
//	//const float velocity = 2;
//
//	glm::vec3 new_heading(0);
//	glm::vec3 avg_heading(0);
//	glm::vec3 avg_position(0);
//	float neighbors = 0;
//
//	for (std::size_t j{ 0 }; j < N; ++j) {
//		//debug[j] = distances[j];
//		if (j == i) { continue; }
//
//		if (distances[j] < range) {
//			++neighbors;
//
//			//Influence from separation: normalized direction scaled by distance and force
//			if (distances[j] != 0) { new_heading += glm::normalize(boids[i].position - boids[j].position) * f_separation / distances[j]; }
//			//debug[j] = distances[j];
//
//			avg_heading += boids[j].heading - avg_heading / neighbors;
//
//			avg_position += boids[j].position - avg_position / neighbors;
//		}
//	}
//	glm::vec3 test_pos(glm::normalize(boids[i].position - boids[1].position) * f_separation / distances[1]);
//	//glm::vec3 test_pos = glm::vec3(0);
//
//	new_heading += avg_heading * f_alignment;
//	new_heading += (avg_position - boids[i].position) * f_cohesion;
//	new_heading = glm::normalize(new_heading);
//
//	new_boids[i] = Boid{ boids[i].position + new_heading * velocity, new_heading };
//	//new_boids[i] = Boid{ test_pos, boids[i].position + new_heading * velocity};
//}

__global__
void update(const Boid* boids, Boid* new_boids)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (std::size_t i{ index }; i < N; i += stride)
	{
		glm::vec3 new_heading(0);
		glm::vec3 avg_heading(0);
		glm::vec3 avg_position(0);
		float neighbors = 0;
		for (std::size_t j{ 0 }; j < N; ++j)
		{
			if (j == i) { continue; }

			float dis = glm::distance(boids[i].position, boids[j].position);
			
			if (dis < range && dis != 0)
			{
				++neighbors;

				new_heading += (boids[i].position - boids[j].position) * f_separation;

				avg_heading += boids[j].heading - avg_heading / neighbors;

				avg_position += boids[j].position - avg_position / neighbors;
			}
		}

		new_heading += avg_heading * f_alignment;
		new_heading += (avg_position - boids[i].position) * f_cohesion;
		new_heading = glm::normalize(new_heading);

		new_boids[i] = Boid{ boids[i].position + new_heading * velocity, new_heading };
	}
}

int main()
{
	//auto seed = std::random_device{}();
	//std::mt19937 mt{ seed };
	std::mt19937 mt{ 1 };
	std::uniform_int_distribution<> dist_pos{ 0, 200 };
	std::uniform_int_distribution<> dist_head{ -1, 1 };

	//std::vector<Boid> boids(N);
	//Boid* boids = new Boid[N];
	Boid* boids;
	Boid* new_boids;
	cudaMallocManaged(&boids, N * sizeof(Boid));
	cudaMallocManaged(&new_boids, N * sizeof(Boid));


	for (std::size_t i{ 0 }; i < N; ++i)
	{
		boids[i].position = glm::vec3({dist_pos(mt), dist_pos(mt), dist_pos(mt)});
		boids[i].heading = glm::normalize(glm::vec3(1));
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (std::size_t i{ 0 }; i < 1; ++i)
	{
		update<<<1, 256>>>(boids, new_boids);
		cudaDeviceSynchronize();

		std::swap(boids, new_boids);
		std::cout << "Position: " << boids[0].position.x << ", " << boids[0].position.y << ", " << boids[0].position.z << std::endl;
		std::cout << "Heading: " << boids[0].heading.x << ", " << boids[0].heading.y << ", " << boids[0].heading.z << std::endl;

	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Performance: " << duration.count() << "ms" << std::endl;

	//std::cout << "Heading: " << boids[0].heading.x << ", " << boids[0].heading.y << ", " << boids[0].heading.z << std::endl;
	//std::cout << "Position: " << boids[0].position.x << ", " << boids[0].position.y << ", " << boids[0].position.z << std::endl;

	cudaFree(boids);
	cudaFree(new_boids);

	return 0;
}
