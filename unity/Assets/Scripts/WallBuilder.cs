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
    private static Vector3 startCoordinate = new Vector3(0, 1f + scaleY / 2, 10f);

    private List<GameObject> _wallTiles = new List<GameObject>();

    public void RemoveWall()
    {
        foreach (var tile in _wallTiles) {
            Destroy(tile);
        }
        _wallTiles = new List<GameObject>();
    }

    public void BuildWall(List<List<bool>> wall)
    {
        RemoveWall();
        float startX = -scaleX * wall[0].Count / 2 + scaleX / 2;
        Vector3 pos = startCoordinate;
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
