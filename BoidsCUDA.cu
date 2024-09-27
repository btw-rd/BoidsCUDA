#define GLM_FORCE_CUDA
#include <iostream>
#include <chrono>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm.hpp>

#define f_separation 1.0f
#define f_alignment 1.0f
#define f_cohesion 1.0f
#define range 5.0f
#define velocity 2.0f

struct Boid {
	glm::vec3 position{};
	glm::vec3 heading{};
};

__constant__ constexpr int N = 1 << 13;

__global__
void update(const Boid* boids, Boid* new_boids)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

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

				// Separation force per neighbor
				new_heading += (boids[i].position - boids[j].position) * f_separation;

				// Alignment accumulation per neighbor
				avg_heading += boids[j].heading;

				// Cohesion accumulation per neighbor
				avg_position += boids[j].position;
			}
		}

		// Alignment averaging and scaling
		new_heading += avg_heading * f_alignment / neighbors;
		// Cohesion alignment and scaling
		new_heading += (avg_position / neighbors - boids[i].position) * f_cohesion;

		new_heading = glm::normalize(new_heading);

		new_boids[i] = Boid{ boids[i].position + new_heading * velocity, new_heading };
	}
}

int main()
{
	//auto seed = std::random_device{}();
	//std::mt19937 mt{ seed };
	std::mt19937 mt{ 1 };
	std::uniform_int_distribution<> dist_pos{ 0, 20 };
	std::uniform_int_distribution<> dist_head{ -1, 1 };


	// Allocate memory for boids arrays on the host and the device
	//Boid* boids = new Boid[N];
	Boid* h_boids;
	Boid* h_new_boids;

	size_t bytes = N * sizeof(Boid);
	h_boids = (Boid*)malloc(bytes);
	h_new_boids = (Boid*)malloc(bytes);

	Boid* d_boids;
	Boid* d_new_boids;

	cudaMalloc(&d_boids, bytes);
	cudaMalloc(&d_new_boids, bytes);

	//cudaMallocManaged(&boids, N * sizeof(Boid));
	//cudaMallocManaged(&new_boids, N * sizeof(Boid));

	int blockSize = 512;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// Initialize host boids
	for (std::size_t i{ 0 }; i < N; ++i)
	{
		h_boids[i].position = glm::vec3({dist_pos(mt), dist_pos(mt), dist_pos(mt)});
		h_boids[i].heading = glm::normalize(glm::vec3(1));
	}


	for (std::size_t i{ 0 }; i < 10; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();

		// `new_boids` will never be read on device so let's save this expensive move operation and only move `boids`
		cudaMemcpy(d_boids, h_boids, bytes, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_new_boids, h_new_boids, bytes, cudaMemcpyHostToDevice);

		update<<<numBlocks, blockSize>>>(d_boids, d_new_boids);
		cudaDeviceSynchronize();

		// As above, but reverse: the values in `boids` are junk and are just going to be overwritten on the next update
		//cudaMemcpy(h_boids, d_boids, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_new_boids, d_new_boids, bytes, cudaMemcpyDeviceToHost);

		std::swap(h_boids, h_new_boids);

		//std::cout << "Position: " << h_boids[0].position.x << ", " << h_boids[0].position.y << ", " << h_boids[0].position.z << std::endl;
		//std::cout << "Heading: " << h_boids[0].heading.x << ", " << h_boids[0].heading.y << ", " << h_boids[0].heading.z << std::endl;

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Performance: " << duration.count() << "ms" << std::endl;
	}


	//std::cout << "Heading: " << boids[0].heading.x << ", " << boids[0].heading.y << ", " << boids[0].heading.z << std::endl;
	//std::cout << "Position: " << boids[0].position.x << ", " << boids[0].position.y << ", " << boids[0].position.z << std::endl;

	cudaFree(d_boids);
	cudaFree(d_new_boids);

	free(h_boids);
	free(h_new_boids);

	return 0;
}
