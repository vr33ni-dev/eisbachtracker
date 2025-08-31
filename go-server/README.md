# 🌊 EisbachTracker PWA

## Backend (Go)

The Go server downloads a CSV file from gkd.bayern.de, unzips it, and extracts the latest water temperature.  
It uses APIs to get current weather and water data. (Water level & flow)
It also scrapes the latest historical water levels from the official Hochwassernachrichtendienst Bayern site.

Powered by:

- Go (1.24)
- PostgreSQL (Neon in production)
- Flyway for DB migrations
- Render for hosting
- Makefile for local DX

---

## Local Development Setup

### Prerequisites

- Go installed (>= 1.24)
- PostgreSQL running locally (``brew services start postgresql``)
- Flyway installed → <https://flywaydb.org/download>

---

### Local .env for Go

```cmd
//.env
//see .env.example for reference
ENV=local
DATABASE_URL=postgres://your-username@localhost:5432/eisbach 
EGELALARM_API_URL=Pegelalarm API
HND_BAYERN_URL=Hochwassernachrichtendienst Bayern
```

---

### Flyway Config for Local

```cmd
//flyway.conf
flyway.url=jdbc:postgresql://localhost:5432/eisbach?sslmode=disable flyway.user=your-username 
flyway.password= 
flyway.schemas=public 
flyway.locations=filesystem:./db/migrations
```

---

## Useful Make Commands (local only)

|Command|What it does|
|-------|------------|
|`make run`|Run Go server locally|
|`make migrate-local`|Apply local DB migrations|
|`make reset-local`|Drop & recreate local DB & run migrations|
|`make flyway-info-local`|Show local migration status|
|`make flyway-repair-local`|Repair Flyway checksums locally|

---

## API Endpoints

|Endpoint|Method|Description|
|--------|------|-----------|
|`/api/surfers`|GET|Get all surfer entries|
|`/api/surfers`|POST|Add new surfer entry|
|`/api/surfers/predict`|GET|Predict surfer count|
|`/api/conditions/weather`|GET|Get latest weather conditions|
|`/api/conditions/water/temperature`|GET|Get latest water temperature|
|`/api/conditions/water/history`|GET|Get historical data on water level and flow|
|`/api/conditions/water`|GET|Get latest water level and flow|

---

## Production Deploy (Render)

Production = Dockerized Go server running on Render  
DB = Neon Postgres Cloud Database

---

### Dockerfile (multi-stage)

- Compiles Go binary
- Installs Flyway
- Runs `flyway migrate` automatically on container start
- Runs Go app

---

### Environment Variables on Render

|Key|Value|
|---|-----|
|DATABASE_URL|Postgres URL for Go app (Neon)|
|PEGELALARM_API_URL|Pegelalarm API|
|HND_BAYERN_URL|Hochwassernachrichtendienst Bayern Website|
|FLYWAY_URL|JDBC URL for Flyway (Neon)|
|FLYWAY_USER|neondb_owner|
|FLYWAY_PASSWORD|password|
|ENV|production|

---

### Render Settings

|Field|Value|
|-----|-----|
|Root Directory|`go-server`|
|Build Command|`docker build -t eisbach .`|
|Start Command|leave empty|

---

## Prod Migrations (manual trigger)

If you add a new migration file (like `V3__add_column.sql`):

Run locally:

```bash
make migrate-prod
```

(This applies migrations to Neon DB)

---

## Prediction logic

### Prediction logic diagram - flow

        +--------------------+
        | User submits count |
        +--------------------+
                    |
                    v
        +---------------------------------------+
        | Fetch current conditions              |
        | - Air Temp (OpenWeather)              |
        | - Water Temp (API)                    |
        | - Weather Condition  (OpenWeather)    |
        +---------------------------------------+
                    |
                    v
        +-----------------------------+
        | Apply Prediction Factors    |
        |-----------------------------|
        | - Air Temp Weight           |
        | - Water Temp Weight         |
        | - Weather Condition Weight  |
        | - Time of Day Boost         |
        | - Weekend Boost             |
        +-----------------------------+
                    |
                    v
        +-------------------------+
        | Final Surfer Prediction |
        +-------------------------+

### Prediction logic diagram - weighting of factors

          | Current Time     |  ---> hour = now.getHours()
          +------------------+
                      |
                      v
          +---------------------------+
          | basePredictionByHour(hour)| ← pulls historical avg from DB
          |     e.g. hour 6 = 4.2     |
          +---------------------------+
                      |
                      v
      +----------------------------------------------+
      | calculateFactor(hour, temp, weather, level)  |
      |                                              |
      | 🕒 Time of Day      → modifies + / -         |
      | ❄️ Water Temp       → modifies + / -         |
      | 🌧️ Weather           → modifies -             |
      | 🌊 Water Level       → modifies + / -         |
      +----------------------------------------------+
                      |
                      v
          +-----------------------------+
          | Prediction = base * factor  |
          | e.g. 4.2 * 0.8 = 3.36 → 3   |
          +-----------------------------+
                      |
                      v
               🎯 Final Prediction
