#pragma once
#include "PointSet.h"


namespace PhysIKA
{
	template<typename DataType3f>
	class UnstructuredPointSet : public PointSet<DataType3f>
	{
	public:
		UnstructuredPointSet();
		~UnstructuredPointSet();

	private:
	};


#ifdef PRECISION_FLOAT
	template class UnstructuredPointSet<DataType3f>;
#else
	template class UnstructuredPointSet<DataType3d>;
#endif
}

