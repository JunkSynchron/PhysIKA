/*
* @file cuda_matrix_3x3_test.cpp
* @brief cuda test for Matrix<Scalar, 3>.
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include <iostream>
#include "Physika_Core/Matrices/Test/matrix_3x3_test.h"
using namespace Physika;

int main()
{
    testMatrix3x3();
    std::system("pause");
    return 0;
}