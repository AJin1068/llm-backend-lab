import asyncio
import time


async def good(i):
    # 루프에 양보하며 1초 대기
    await asyncio.sleep(1)
    print(f"good {i} 끝")


async def bad(i):
    time.sleep(1)
    print(f"bad {i} 끝")


async def bad2(i):
    await asyncio.to_thread(time.sleep, 1)  # time.sleep(1)을 별도 스레드에서 실행
    print(f"bad2 {i} 끝")


async def main():
    start = time.perf_counter()
    await asyncio.gather(good(1), good(2), good(3))  # 3개를 동시에 실행
    print(f"good 총 {time.perf_counter() - start:.1f}초")

    start = time.perf_counter()
    await asyncio.gather(bad(1), bad(2), bad(3))
    print(f"bad 총 {time.perf_counter() - start:.1f}초")

    start = time.perf_counter()
    await asyncio.gather(bad2(1), bad2(2), bad2(3))
    print(f"bad2 총 {time.perf_counter() - start:.1f}초")


asyncio.run(main())  # 이벤트 루프를 만들고 main()을 실행
