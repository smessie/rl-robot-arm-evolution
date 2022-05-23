using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallBuilder : MonoBehaviour
{
    public GameObject wallTilePrefab;

    private static float scaleX = 2f;
    private static float scaleY = 2f;
    private static float scaleZ = 1f;
    private static float startY = 1f+scaleY/2;
    private static float startZ = 10f;

    private static int wallAmount = 0;
    private float distanceBetweenWalls = 6;

    private List<GameObject> _wallTiles = new List<GameObject>();

    public void clearWalls()
    {
        foreach (var tile in _wallTiles) {
            Destroy(tile);
        }
        _wallTiles = new List<GameObject>();
        wallAmount = 0;
    }

    public void BuildWall(List<List<bool>> wall)
    {
        float startX = -scaleX * wall[0].Count / 2 + scaleX / 2;
        Vector3 pos = new Vector3(startX, startY, startZ + wallAmount*distanceBetweenWalls);
        for (int r = wall.Count-1; r >= 0; r--) {
            pos.x = startX;
            for (int c = 0; c < wall[0].Count; c++) {
                if (wall[r][c]) {
                    AddWallTile(pos);
                }
                pos.x += scaleX;
            }
            pos.y += scaleY;
        }
    }

    private void AddWallTile(Vector3 pos)
    {
        GameObject wallTile = Instantiate(
            wallTilePrefab, // type GameObject we want to make
            pos, // Position on where we want to instantiate it
            Quaternion.identity // Turn/rotation
        );
        _wallTiles.Add(wallTile);
        wallTile.transform.localScale = new Vector3(scaleX, scaleY, scaleZ); // Multiply by 2 because we only show half of module
    }
}
