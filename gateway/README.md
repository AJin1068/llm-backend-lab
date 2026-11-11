# gateway

Kotlin(Spring) LLM 게이트웨이 자리입니다. 11/23(PR 16)부터 구현합니다.

계획: 중계와 스트리밍, 호출별 토큰·비용 기록, Redis 레이트 리밋, Resilience4j 재시도·타임아웃·서킷 브레이커·폴백.
`agent/`는 LLM 호출 주소(base_url)를 설정으로 바꿔 이 게이트웨이를 거치게 됩니다.
