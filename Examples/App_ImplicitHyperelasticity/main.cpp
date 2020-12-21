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

#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule_test.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"


using namespace std;
using namespace PhysIKA;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> child3 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child3);

	child3->varHorizon()->setValue(0.0125);

	auto ptRender1 = std::make_shared<PointRenderModule>();
	ptRender1->setColor(Vector3f(0, 1, 1));
	child3->addVisualModule(ptRender1);

	child3->setMass(1.0);
	Vector3f center(0.0, 0.0, 0.0);
	Vector3f rectangle(0.06, 0.05, 0.05);
	child3->loadParticles(center- rectangle, center + rectangle, 0.005);
	
	child3->translate(Vector3f(0.5, 0.2, 0.5));
	child3->setVisible(true);
	auto hyper = std::make_shared<HyperelasticityModule_test<DataType3f>>();
	hyper->x_border = 0.5;
	hyper->release_adjust_points_reachTargetPlace = true;
	//hyper->setEnergyFunction(HyperelasticityModule<DataType3f>::Quadratic);
	child3->setElasticitySolver(hyper);
	child3->getElasticitySolver()->setIterationNumber(10);
}


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


