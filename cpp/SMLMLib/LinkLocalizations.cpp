// Localization clustering and linking
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "DLLMacros.h"
#include "Vector.h"
#include "ThreadUtils.h"
#include "KDTree.h"
#include "StringUtils.h"


typedef int (*ClusterLocsCallback)(const int* mapping, const float* centers);

/*
* Clustering:
 */
template<int D>
int ClusterLocs_(float* pos_, int* spotCluster, const float* distance_, int numspots, ClusterLocsCallback callback)
{
	typedef Vector<float, D> Pt;

	Pt distance = *(const Pt*)distance_;
	Pt invDist = 1.0f / distance;

	std::vector <Pt> pos((Pt*)pos_, (Pt*)pos_ + numspots);
	Pt* result = (Pt*)pos_;

	std::vector<Pt> clusterPos;
	std::vector<int> clusterCounts;

	std::fill(spotCluster, spotCluster + numspots, -1);
	KDTree<float, D> tree(pos, 20);

	// create new clusters for unassigned spots
	auto createNewClusters = [&]() {
		int removed = 0, created=0;
		for (int i = 0; i < numspots; i++) {
			if (spotCluster[i] >= 0)
			{
				int cl = spotCluster[i];

				if (((clusterPos[cl] - pos[i]) * invDist).sqLength() < 1.0f)
					continue;
				else {
					removed++;
					clusterCounts[cl]--;
				}
			}

			int clusterIdx = (int)clusterPos.size();
			clusterPos.push_back(pos[i]);
			clusterCounts.push_back(1);
			spotCluster[i] = clusterIdx;

			created++;

			std::vector<int> nb;
			tree.AddPointsInEllipsoidToList(pos[i], distance, nb, 0);

			for (int j : nb) {
				if (spotCluster[j] < 0)
					spotCluster[j] = clusterIdx;
			}
		}
		DebugPrintf("Removed %d spots from clusters, created %d new clusters\n", removed, created);
		return created;
	};

	auto updateClusterMeans = [&]() {
		std::vector<int> newClusterIdx(clusterCounts.size());
		int newClusters = 0;
		for (int i = 0; i < clusterCounts.size();i++) {
			newClusterIdx[i] = newClusters;
			if (clusterCounts[i] > 0) {
				newClusters++;
			}
		}
		for (int i = 0; i < numspots; i++) {
			spotCluster[i] = newClusterIdx[spotCluster[i]];
		}
		//DebugPrintf("Removed %d/%d empty clusters\n", (clusterCounts.size() - newClusters), clusterCounts.size());
		clusterCounts.resize(newClusters);
		clusterPos.resize(newClusters);

		std::fill(clusterPos.begin(), clusterPos.end(), Pt{});
		std::fill(clusterCounts.begin(), clusterCounts.end(), 0);
		for (int i = 0; i < numspots; i++) {
			clusterCounts[spotCluster[i]]++;
			clusterPos[spotCluster[i]] += pos[i];
		}
		for (int i = 0; i < clusterCounts.size(); i++) {
			clusterPos[i] /= clusterCounts[i];
		}

		return (int)clusterCounts.size();
	};

	auto update = [&]() {
		bool modified = false;
		for (int cl = 0; cl < clusterPos.size(); cl++) {
			if (clusterCounts[cl] == 0)
				continue;

			std::vector<int> nb;
			tree.AddPointsInEllipsoidToList(clusterPos[cl], distance, nb, 0);

			// see if this cluster is a better fit than the current one
			for (int j : nb) {
				if (spotCluster[j] == cl)
					continue;

				// steal it	
				int clusterJ = spotCluster[j];
				if (clusterCounts[cl] > clusterCounts[clusterJ] || clusterCounts[clusterJ]==1) {
					spotCluster[j] = cl;
					clusterCounts[cl]++;
					clusterCounts[clusterJ] --;
					modified = true;
				}
				else if (clusterCounts[cl] == clusterCounts[clusterJ]) // tie breaker
				{
					float dist2_j = (pos[j] - clusterPos[clusterJ]).sqLength();
					float dist2 = (pos[j] - clusterPos[cl]).sqLength();

					if (dist2 < dist2_j) {
						spotCluster[j] = cl;
						clusterCounts[cl]++;
						clusterCounts[clusterJ] --;
						modified = true;
					}
				}
			}
		}

		return modified;
	};

	int it = 0;
	while (true) {
		int created=createNewClusters();
		int n = updateClusterMeans();
		bool modified=update() || created!=0;
		DebugPrintf("Iteration %d. Number of clusters: %d\n", it, n);
		it++;

		if (callback)
			callback(spotCluster, (const float*)clusterPos.data());

		if (!modified)
			break;

	}
	updateClusterMeans();


	Pt* dst = (Pt*)pos_;
	for (int i = 0; i < clusterPos.size(); i++) {
		dst[i] = clusterPos[i];
	}
	return (int)clusterPos.size();
}



// modifies in place
CDLL_EXPORT int ClusterLocs(int dims, float* pos, int* mappingToNew, const float* distance, int numspots, ClusterLocsCallback callback)
{
	if (dims == 2) {
		return ClusterLocs_<2>(pos, mappingToNew, distance, numspots, callback);
	}
	if (dims == 3) {
		return ClusterLocs_<3>(pos, mappingToNew, distance, numspots, callback);
	}
	DebugPrintf("ClusterLocs: %d is invalid number of dimensions.\n");
	return -1;
}

typedef int (*FindNeighborCallback)(int startA, int numspotsA, const int* counts, const int* indicesB, int numIndicesB);

template<int D>
void FindNeighbors_(int numspotsA, const float* coordsA, int numspotsB, const float* coordsB, const float* maxDistance, int minBatchSize,
	FindNeighborCallback cb)
{
	typedef Vector<float, D> Pt;
	std::vector<Pt> pointsB(((const Pt*)coordsB), ((const Pt*)coordsB) + numspotsB);
	std::vector<Pt> pointsA(((const Pt*)coordsA), ((const Pt*)coordsA) + numspotsA);

	Pt maxdist = *(const Pt*)maxDistance;

	KDTree<float, D> tree(pointsB, 20);
	IterateThroughNeighbors(tree, pointsA, maxdist, minBatchSize, 0,
		[&](int processedUpto, std::vector<int> indices, std::vector<int> startpos, std::vector<int> counts) {
			cb(processedUpto, (int)counts.size(), counts.data(), indices.data(), (int)indices.size());
			return true;
		});
}


// Find spots from B that are near spots from A
CDLL_EXPORT int FindNeighbors(int numspotsA, const float* coordsA, int numspotsB, const float* coordsB, int dims, const float* maxDistance, int minBatchSize,
	FindNeighborCallback cb)
{
	switch (dims) {
	case 1:
		FindNeighbors_<1>(numspotsA, coordsA, numspotsB, coordsB, maxDistance, minBatchSize, cb);
		break;
	case 2:
		FindNeighbors_<2>(numspotsA, coordsA, numspotsB, coordsB, maxDistance, minBatchSize, cb);
		break;
	case 3:
		FindNeighbors_<3>(numspotsA, coordsA, numspotsB, coordsB, maxDistance, minBatchSize, cb);
		break;
	case 4:
		FindNeighbors_<4>(numspotsA, coordsA, numspotsB, coordsB, maxDistance, minBatchSize, cb);
		break;
	}
	return 0;
}


CDLL_EXPORT int LinkLocalizations(int numspots, int* framenum, Vector3f* xyI, Vector3f* crlbXYI, float maxDist, float maxIntensityDist, int frameskip, 
									int *linkedSpots, int* startframes, int *framecounts, Vector3f* linkedXYI, Vector3f* linkedcrlbXYI)
{
	std::vector<Vector3f> linkedMeans (numspots), linkedVar(numspots);

	// Find number of framenum
	int nframes = 0;
	for (int i = 0; i < numspots; i++) {
		if (framenum[i] >= nframes) nframes = framenum[i] + 1;
	}

	// Organise by frame number
	std::vector<std::vector<int>> frameSpots (nframes);
	for (int i = 0; i < numspots; i++)
		frameSpots[framenum[i]].push_back(i);

	// Clear linked spots
	for (int i = 0; i < numspots; i++)
		linkedSpots[i] = -1;

	int nlinked = 0;
	auto linkspots = [&](int prev, int b) {
		// prev may or may not already be linked. b is definitely unlinked
		if (linkedSpots[prev] < 0) {
			linkedMeans[nlinked] = xyI[prev];
			linkedVar[nlinked] = crlbXYI[prev] * crlbXYI[prev];
			linkedSpots[prev] = nlinked++;
		}
		int j = linkedSpots[prev];
		linkedSpots[b] = j;

		Vector3f varB = crlbXYI[b]*crlbXYI[b];
		Vector3f totalVar = 1.0f / (1.0f / linkedVar[j] + 1.0f / varB);
		linkedMeans[j] = totalVar * (linkedMeans[j] / linkedVar[j] + xyI[b] / varB);
		linkedVar[j] = totalVar;
	};

	// Connect spots
	for (int f = 1; f < nframes; f++) {
		for (int b = std::max(0, f - frameskip-1); b < f; b++) {

			for (int i : frameSpots[f])
			{
				for (int prev : frameSpots[b]) {

					Vector3f cmpPos;
					if (linkedSpots[prev] >= 0)
						cmpPos = linkedMeans[linkedSpots[prev]];
					else
						cmpPos = xyI[prev];

					Vector2f diff = Vector2f{ xyI[i][0],xyI[i][1] } -Vector2f{ cmpPos[0],cmpPos[1] };
					float xydist = diff.length();
					float Idist = abs(xyI[i][2] - cmpPos[2]);

					// find spots within maxDist
					float maxPixelDist = maxDist;// *Vector2f{ crlbXYI[i][0],crlbXYI[i][1] }.length();
					if (xydist < maxPixelDist)///&& 
//						Idist < maxIntensityDist * crlbXYI[i][2])
					{
						linkspots(prev, i); // i should be linked to prev
						break;
					}
				}
			}

		}
	}

	// Give all non-linked spots a unique id
	for (int i = 0; i < numspots; i++)
	{
		if (linkedSpots[i] < 0)
			linkedSpots[i] = nlinked++;
	}

	std::vector<int> endframes(nlinked,-1);
	// Compute startframes/framecounts
	for (int i = 0; i < nlinked; i++) {
		startframes[i] = -1;
	}
	for (int i = 0; i < numspots; i++)
	{
		int j = linkedSpots[i];
		if (startframes[j]<0 || startframes[j]>framenum[i])
			startframes[j] = framenum[i];
		if (endframes[j] < 0 || endframes[j] < framenum[i])
			endframes[j] = framenum[i];
	}

	for (int i = 0; i < nlinked; i++)
		framecounts[i] = endframes[i] - startframes[i] + 1;

	// Compute spot mean xy and summed intensity
	for (int i = 0; i < nlinked; i++) {
		linkedXYI[i] = {};
		linkedXYI[i][0] = linkedMeans[i][0];
		linkedXYI[i][1] = linkedMeans[i][1];
		linkedcrlbXYI[i] = linkedVar[i].sqrt();
	}

	for (int i = 0; i < numspots; i++)
	{
		int j = linkedSpots[i];
		linkedXYI[j][2] += xyI[i][2];
	}

	return nlinked;
}


