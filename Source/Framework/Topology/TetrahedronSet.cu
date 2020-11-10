#include "TetrahedronSet.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Core/Utility.h"

namespace PhysIKA
{
	template<typename TDataType>
	TetrahedronSet<TDataType>::TetrahedronSet()
		: TriangleSet<TDataType>()
	{
		
	}

	template<typename TDataType>
	TetrahedronSet<TDataType>::~TetrahedronSet()
	{
	}


	template<typename TDataType>
	bool TetrahedronSet<TDataType>::initializeImpl()
	{
		
		return true;
	}


	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons)
	{
		m_tethedrons.resize(tetrahedrons.size());
		Function1Pt::copy(m_tethedrons, tetrahedrons);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(DeviceArray<Tetrahedron>& tetrahedrons)
	{
		if (tetrahedrons.size() != m_tethedrons.size())
		{
			m_tethedrons.resize(tetrahedrons.size());
		}

		Function1Pt::copy(m_tethedrons, tetrahedrons);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::loadTetFile(std::string filename)
	{
		std::string filename_node = filename;	filename_node.append(".node");
		std::string filename_ele = filename;	filename_ele.append(".ele");

		std::ifstream infile_node(filename_node);
		std::ifstream infile_ele(filename_ele);
		if (!infile_node || !infile_ele) {
			std::cerr << "Failed to open the tetrahedron file. Terminating.\n";
			exit(-1);
		}

		std::string line;
		std::getline(infile_node, line);
		std::stringstream ss_node(line);

		int node_num;
		ss_node >> node_num;
		std::vector<Coord> nodes;
		for (int i = 0; i < node_num; i++)
		{
			std::getline(infile_node, line);
			std::stringstream data(line);
			int id;
			Coord v;
			data >> id >> v[0] >> v[1] >> v[2];
			nodes.push_back(v);
		}

		
		std::getline(infile_ele, line);
		std::stringstream ss_ele(line);

		int ele_num;
		ss_ele >> ele_num;
		std::vector<Triangle> tris;
		std::vector<Tetrahedron> tets;
		for (int i = 0; i < ele_num; i++)
		{
			std::getline(infile_ele, line);
			std::stringstream data(line);
			int id;
			Tetrahedron tet;
			data >> id >> tet[0] >> tet[1] >> tet[2] >> tet[3];
			tet[0] -= 1;
			tet[1] -= 1;
			tet[2] -= 1;
			tet[3] -= 1;
			tets.push_back(tet);

			tris.push_back(Triangle(tet[0], tet[1], tet[2]));
			tris.push_back(Triangle(tet[0], tet[3], tet[1]));
			tris.push_back(Triangle(tet[1], tet[3], tet[2]));
			tris.push_back(Triangle(tet[0], tet[2], tet[3]));
		}

		this->setPoints(nodes);

		this->setTriangles(tris);
		this->setTetrahedrons(tets);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::getVolume(DeviceArray<Real>& volume)
	{

	}
}