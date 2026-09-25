# Meeting Notes — 2026-09-10

Attendees: me, team lead, infra engineer.

## Decisions
- We decided to use **Redis** for the response cache instead of an
  in-memory Python dict, since the service needs to survive restarts.
- Approved budget for the GPU rental: **$4,300** for the next quarter.
- Next review scheduled for the third week of October.

## Action items
- Write a migration script for the cache layer.
- Benchmark Redis latency vs. current in-memory cache before switching.
