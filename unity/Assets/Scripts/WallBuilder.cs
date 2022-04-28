using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallBuilder : MonoBehaviour
{
    public GameObject wallTilePrefab;

    private Vector3 startCoordinate = new Vector3(-6.5f, 1f, 10f);
    private float scaleX = 1f;
    private float scaleY = 1f;
    private float scaleZ = 0.3f;

    private List<GameObject> _wallTiles = new List<GameObject>();

    public void RemoveWall()
    {
        foreach (var tile in _wallTiles) {
            Destroy(tile);
        }
    }

    public void BuildWall(List<List<bool>> wall)
    {
        RemoveWall();
        Vector3 pos = startCoordinate;
        for (int r = wall.Count-1; r >= 0; r--) {
            for (int c = 0; c < wall[0].Count; c++) {
                if (wall[r][c]) {
                    AddWallTile(pos + new Vector3(scaleX/2, scaleY/2, 0f));
                }
                pos.x += scaleX;
            }
            pos.x = startCoordinate.x;
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
