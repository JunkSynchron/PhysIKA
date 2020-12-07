#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Topology/TetrahedronSet.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/ParticleSystem/HyperelasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule_test.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"


using namespace std;
using namespace PhysIKA;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	std::shared_ptr<HyperelasticBody<DataType3f>> elasticObj = std::make_shared<HyperelasticBody<DataType3f>>();
	root->addChild(elasticObj);

	auto pointRender = std::make_shared<PointRenderModule>();
	pointRender->setColor(Vector3f(0, 1, 1));
	elasticObj->addVisualModule(pointRender);

	elasticObj->setMass(1.0);

	Vector3f center(0.0, 0.0, 0.0);
	Vector3f rectangle(0.06, 0.05, 0.05);
	//elasticObj->loadParticles(center- rectangle, center + rectangle, 0.005);
	elasticObj->loadFile("../../Media/smesh/armadillo-coarse.smesh");
	elasticObj->scale(0.1);
	elasticObj->translate(Vector3f(0.5f, 0.25f, 0.5f));

	double x_border = 0.5;
	elasticObj->setVisible(true);

	//elasticObj->scale(0.1);

	elasticObj->getMeshNode()->setActive(false);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	elasticObj->getMeshNode()->addVisualModule(meshRender);
}


int main()
{
	CreateScene();

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


