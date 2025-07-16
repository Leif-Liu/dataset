class Solution:
    def findTheCity(
        self, n: int, edges: List[List[int]], distanceThreshold: int
    ) -> int:
        dist = [[1 << 31 for _ in range(n)] for _ in range(n)]
        for x, y, d in edges:
            dist[x][y] = d
            dist[y][x] = d
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        count = [len(list(filter(lambda x: x <= distanceThreshold, x))) for x in dist]
        target = min(count)
        for i in reversed(range(n)):
            if count[i] == target:
                return i