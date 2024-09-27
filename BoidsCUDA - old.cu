#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm.hpp>

struct Boid {
	glm::vec3 position{};
	glm::vec3 heading{};
};

__device__ __constant__ const float f_separation = 1;
__device__ __constant__ const float f_alignment = 1;
__device__ __constant__ const float f_cohesion = 1;
__device__ __constant__ const float range = 10;
__device__ __constant__ const float velocity = 2;

__constant__ constexpr int N = 1 << 5;

__global__
void child_update(std::size_t i, const Boid* boids, float* distances)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (std::size_t j{ index }; j < N; j += stride)
	{
		if (j == i) { continue; }

		distances[j] = glm::distance(boids[i].position, boids[j].position);
	}
}

__global__
void update(const Boid* boids, Boid* new_boids)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	//for (std::size_t i { 0 }; i < N; ++i)
	for (std::size_t i{ index }; i < N; i += stride)
	{
		glm::vec3 new_heading(0);
		glm::vec3 avg_heading(0);
		glm::vec3 avg_position(0);
		float neighbors = 0;

		//float* distances;
		//cudaMallocManaged(&distances, N * sizeof(float));
		float distances[N];

		child_update<<<1, 256 >>>(i, boids, distances);
		//cudaDeviceSynchronize();

		for (std::size_t j{ 0 }; j < N; ++j) {
			if (j == i) { continue; }

			//float dis = glm::distance(boids[i].position, boids[j].position);

			if (distances[j] < range) {
				++neighbors;

				//Influence from separation: normalized direction scaled by distance and force
				new_heading += glm::normalize(boids[i].position - boids[j].position) * f_separation / distances[j];

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

	//std::mt19937 mt{ std::random_device{}() };
	std::mt19937 mt{ 1 };
	std::uniform_int_distribution<> dist_pos{ 0, 20 };
	std::uniform_int_distribution<> dist_head{ -1, 1 };

	//std::vector<Boid> boids(N);
	//Boid* boids = new Boid[N];
	Boid* boids;
	cudaMallocManaged(&boids, N * sizeof(Boid));

	for (std::size_t i{ 0 }; i < N; ++i)
	{
		boids[i].position = glm::vec3({dist_pos(mt), dist_pos(mt), dist_pos(mt)});
		boids[i].heading = glm::normalize(glm::vec3(1));
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (std::size_t i{ 0 }; i < 10; ++i)
	{
		Boid* new_boids;
		cudaMallocManaged(&new_boids, N * sizeof(Boid));
		
		//update(boids, new_boids);
		update<<<1, 256>>>(boids, new_boids);
		cudaDeviceSynchronize();

		boids = new_boids;
		cudaFree(new_boids);
		//std::cout << "Position: " << boids[0].position.x << ", " << boids[0].position.y << ", " << boids[0].position.z << std::endl;
		//std::cout << "Heading: " << boids[0].heading.x << ", " << boids[0].heading.y << ", " << boids[0].heading.z << std::endl;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Performance: " << duration.count() << "ms" << std::endl;

	//std::cout << "Heading: " << boids[0].heading.x << ", " << boids[0].heading.y << ", " << boids[0].heading.z << std::endl;
	//std::cout << "Position: " << boids[0].position.x << ", " << boids[0].position.y << ", " << boids[0].position.z << std::endl;

	cudaFree(boids);

	return 0;
}
