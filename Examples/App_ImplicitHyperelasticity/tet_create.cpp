#include <iostream>
#include <vector>
#include <fstream>

#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Framework/Topology/TetrahedronSet.h"

std::vector<PhysIKA::DataType3f::Coord> get_particle_cube(PhysIKA::DataType3f::Coord particle_position, PhysIKA::DataType3f::Real distance) {
	typedef PhysIKA::DataType3f::Coord Coord;

	std::vector<PhysIKA::DataType3f::Coord> cube_vertex(8);
	PhysIKA::DataType3f::Real delta[2] = { -distance / 2, distance / 2 };

	cube_vertex[0] = particle_position - Coord(delta[0], delta[1], delta[1]);
	cube_vertex[1] = particle_position - Coord(delta[1], delta[1], delta[1]);
	cube_vertex[2] = particle_position - Coord(delta[1], delta[0], delta[1]);
	cube_vertex[3] = particle_position - Coord(delta[0], delta[0], delta[1]);
	cube_vertex[4] = particle_position - Coord(delta[0], delta[1], delta[0]);
	cube_vertex[5] = particle_position - Coord(delta[1], delta[1], delta[0]);
	cube_vertex[6] = particle_position - Coord(delta[1], delta[0], delta[0]);
	cube_vertex[7] = particle_position - Coord(delta[0], delta[0], delta[0]);

	return cube_vertex;
}

int cube_tetra[5][4] = {
	{0, 1, 4, 3},
	{1, 5, 4, 6},
	{2, 3, 6, 1},
	{3, 7, 6, 4},
	{1, 4, 3, 6},
};

// 一个粒子一个正方体，各个正方体的顶点各自独立，不合并重复顶点； 一个正方体划分成5个四面体: 4个小的 + 1个大的;
bool create_tet_file_from_particles(std::string file_name, 
							std::vector<PhysIKA::DataType3f::Coord> particles, 
							PhysIKA::DataType3f::Real distance) 
{
	typedef PhysIKA::DataType3f::Coord Coord;
	typedef PhysIKA::DataType3f::Real Real;

	std::string filename_node = file_name;	filename_node.append(".node");
	std::string filename_ele = file_name;	filename_ele.append(".ele");

	std::ofstream outfile_node(filename_node);
	std::ofstream outfile_ele(filename_ele);
	if (!outfile_node || !outfile_ele) {
		std::cerr << "Failed to open the tetrahedron file. Terminating.\n";
		return false;
	}

	int total_vertices = particles.size() * 8;
	int total_tets = particles.size() * 5;
	int total_particles = particles.size();
	for (int i = 0; i < total_particles; ++i) {
		int vertex_base_id = i * 8 + 1;
		int element_base_id = i * 5 + 1;
		std::vector<Coord> cube_vertices = get_particle_cube(particles[i], distance);

		for (int j = 0; j < 8; ++j) {
			outfile_node << vertex_base_id + j << ' ';
			outfile_node << cube_vertices[j][0] << ' ' 
						<< cube_vertices[j][1] << ' '
						<< cube_vertices[j][2] << std::endl;
		}

		for (int j = 0; j < 5; ++j) {
			outfile_ele << element_base_id + j << ' ';
			outfile_ele << vertex_base_id + cube_tetra[j][0] << ' '
				<< vertex_base_id + cube_tetra[j][1] << ' '
				<< vertex_base_id + cube_tetra[j][2] << ' '
				<< vertex_base_id + cube_tetra[j][3] << std::endl;
		}
	}
}
