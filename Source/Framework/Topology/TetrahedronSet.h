#pragma once
#include "TriangleSet.h"
#include "Framework/Framework/ModuleTopology.h"


namespace PhysIKA
{
	template<typename TDataType>
	class TetrahedronSet : public TriangleSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;

		TetrahedronSet();
		~TetrahedronSet();

		void loadTetFile(std::string filename);

		void setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons);
		void setTetrahedrons(DeviceArray<Tetrahedron>& tetrahedrons);

		void getVolume(DeviceArray<Real>& volume);

	protected:
		bool initializeImpl() override;

	protected:
		DeviceArray<Tetrahedron> m_tethedrons;
	};

#ifdef PRECISION_FLOAT
	template class TetrahedronSet<DataType3f>;
#else
	template class TetrahedronSet<DataType3d>;
#endif
}

