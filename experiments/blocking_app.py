import asyncio
import time

from fastapi import FastAPI

app = FastAPI()


@app.get("/async-ok")
async def async_ok():
    # 루프에 양보하며 1초 대기
    await asyncio.sleep(1)
    return {"ok": True}


@app.get("/async-bad")
async def async_bad():
    # 루프 스레드 자체를 1초 재움 → 다른 요청도 모두 대기
    time.sleep(1)
    return {"ok": True}


@app.get("/sync-ok")
def sync_ok():  # async가 없는 일반 def
    # 블로킹이지만 FastAPI가 스레드풀에서 실행하므로 루프는 막히지 않음
    time.sleep(1)
    return {"ok": True}
