import asyncio
import sys
import time

import httpx2


async def main(path: str, n: int = 10) -> None:
    async with httpx2.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60) as client:
        start = time.perf_counter()
        await asyncio.gather(*(client.get(path) for _ in range(n)))  # 동시에 n개 요청
        print(f"{path} x{n}: {time.perf_counter() - start:.2f}s")


asyncio.run(main(sys.argv[1]))
