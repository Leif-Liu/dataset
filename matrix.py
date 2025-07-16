from collections import deque


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # 000
        # 010
        # 111
        q = deque()
        M, N = len(mat), len(mat[0])
        ans = [[-1 for _ in range(N)] for _ in range(M)]
        visited = set()
        for i in range(M):
            for j in range(N):
                if mat[i][j] == 0:
                    ans[i][j] = 0
                    q.append((i, j))
                    visited.add((i, j))

        q.append(None)
        distance = 1
        while q:
            top = q.popleft()
            if top is None:
                if q:
                    q.append(None)
                distance += 1
                continue
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                x, y = top[0] + d[0], top[1] + d[1]
                if x < 0 or x >= M or y < 0 or y >= N:
                    continue
                if (x, y) in visited:
                    continue
                ans[x][y] = distance
                visited.add((x, y))
                q.append((x, y))
        return ans