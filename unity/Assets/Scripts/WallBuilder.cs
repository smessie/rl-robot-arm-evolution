using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Script on the main manipulator to build walls and remove them
///
/// The first wall will be built on a certain distance from the anchor.
/// Every subsequent wall will be built on a certain distance from the previous wall
///
/// When removing walls, all walls are removed at one time
///
/// Walls are represented by a 2D array of booleans,
/// true meaning there is a tile on that 'coordinate/index'
/// and false meaning there is not
/// </summary>
public class WallBuilder : MonoBehaviour
{
    /// <summary>
    /// Prefab instantiated by Unity representing one tile of the wall
    /// </summary>
    public GameObject wallTilePrefab;

    // The size of a tile of the wall
    private const float scaleX = 2f;
    private const float scaleY = 2f;
    private const float scaleZ = 1f;

    private const float startY = 1f + (scaleY / 2);
    private const float startZ = 8f;
    private const float distanceBetweenWalls = 5.5f;

    private int wallAmount;
    private List<GameObject> _wallTiles = new List<GameObject>();

    /// <summary>
    /// Remove all walls placed
    /// </summary>
    public void ClearWalls()
    {
        foreach (GameObject tile in _wallTiles)
        {
            Destroy(tile);
        }
        _wallTiles = new List<GameObject>();
        wallAmount = 0;
    }

    /// <summary>
    /// Build one wall. Subsequent walls will be placed at distanceBetweenWalls from this one
    /// </summary>
    /// <param name="wall">2D List of bool representing the wall.
    /// true meaning there is a tile on that 'coordinate/index'
    /// and false meaning there is not</param>
    public void BuildWall(List<List<bool>> wall)
    {
        float startX = (-scaleX * wall[0].Count / 2) + (scaleX / 2);
        Vector3 pos = new Vector3(startX, startY, startZ + (wallAmount * distanceBetweenWalls));
        for (int r = wall.Count - 1; r >= 0; r--)
        {
            pos.x = startX;
            for (int c = 0; c < wall[0].Count; c++)
            {
                if (wall[r][c])
                {
                    AddWallTile(pos);
                }
                pos.x += scaleX;
            }
            pos.y += scaleY;
        }
        wallAmount++;
    }

    /// <summary>
    /// Place one wall tile on a certain position
    /// </summary>
    /// <param name="pos">The position of the tile</param>
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
