package routes

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/vr33ni/eisbachtracker-pwa/go-server/conditions"
	"github.com/vr33ni/eisbachtracker-pwa/go-server/middleware"
	"github.com/vr33ni/eisbachtracker-pwa/go-server/surferdata"
)

// RegisterRoutes registers all API routes
func RegisterRoutes(db *pgxpool.Pool) {
	airService := conditions.NewAirService()
	waterService := conditions.NewWaterService()
	surferService := surferdata.NewService(db, waterService, airService)
	http.HandleFunc("/api/conditions/weather", middleware.WithCORS(handleWeather(airService)))
	http.HandleFunc("/api/conditions/water/temperature", middleware.WithCORS(handleWaterTemperature(waterService)))
	http.HandleFunc("/api/conditions/water/history", middleware.WithCORS(HandleWaterHistory(waterService)))
	http.HandleFunc("/api/conditions/water", middleware.WithCORS(handleWaterLevelAndFlow(waterService)))
	http.HandleFunc("/api/surfers", middleware.WithCORS(handleSurferEntries(surferService)))
	http.HandleFunc("/api/surfers/predict", middleware.WithCORS(handlePrediction(airService, surferService, waterService)))
	http.HandleFunc("/api/ml/health", middleware.WithCORS(handleMLHealth()))
}

// -- Handlers --

func handleWaterTemperature(waterService conditions.WaterDataProvider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		temp, err := waterService.GetLatestWaterTemperature()
		if err != nil {
			log.Println("❌", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"water_temperature": temp,
		})
	}
}

func handleWeather(airService conditions.AirDataProvider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		weatherData, err := airService.GetCurrentWeather()
		if err != nil {
			log.Println("❌", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(weatherData)
	}
}

func handleWaterLevelAndFlow(waterService conditions.WaterDataProvider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		result, err := waterService.GetLatestWaterLevelAndFlow()
		if err != nil {
			log.Printf("❌ Failed to get water level/flow: %v", err)
			http.Error(w, "Failed to get water data", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"water_level":  result.Level,
			"water_flow":   result.Flow,
			"request_date": result.RequestDate,
		})
	}
}

func handlePrediction(airService conditions.AirDataProvider, service *surferdata.Service, waterService conditions.WaterDataProvider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hourStr := r.URL.Query().Get("hour")
		waterTempStr := r.URL.Query().Get("water_temperature")
		airTempStr := r.URL.Query().Get("air_temperature")
		conditionStr := r.URL.Query().Get("weather_condition")

		var hour int
		if hourStr == "" {
			hour = time.Now().Hour()
		} else {
			var err error
			hour, err = strconv.Atoi(hourStr)
			if err != nil {
				http.Error(w, "Invalid hour", http.StatusBadRequest)
				return
			}
		}

		var waterTemp, airTemp *float64
		var weatherCondition *int

		// ✅ Use cached water temp
		if waterTempStr != "" {
			if t, err := strconv.ParseFloat(waterTempStr, 64); err == nil {
				waterTemp = &t
			}
		} else if latest, err := waterService.GetCachedWaterTemperature(); err == nil {
			waterTemp = &latest
		}

		// ✅ Parse airTempStr and conditionStr together
		if airTempStr != "" && conditionStr != "" {
			if t, err := strconv.ParseFloat(airTempStr, 64); err == nil {
				airTemp = &t
			} else {
				http.Error(w, "Invalid air_temperature", http.StatusBadRequest)
				return
			}

			if c, err := strconv.Atoi(conditionStr); err == nil {
				weatherCondition = &c
			} else {
				http.Error(w, "Invalid weather_condition", http.StatusBadRequest)
				return
			}
		} else if current, err := airService.GetCurrentWeather(); err == nil {
			if airTempStr == "" {
				airTemp = &current.Temp
			}
			if conditionStr == "" {
				weatherCondition = &current.Condition
			}
		} else {
			log.Println("⚠️ Could not fetch current weather:", err)
			airTemp = nil
			weatherCondition = nil
		}

		// ✅ Fetch the water level
		var waterLevel float64
		if latestWater, err := waterService.GetLatestWaterLevelAndFlow(); err == nil {
			waterLevel = latestWater.Level
		} else {
			log.Printf("❌ Failed to fetch water level: %v", err)
			waterLevel = 0 // Fallback to 0 if water level cannot be retrieved
		}

		var weatherConditionValue int
		if weatherCondition != nil {
			weatherConditionValue = *weatherCondition
		} else {
			weatherConditionValue = -1 // Default value for unknown weather condition
		}

		prediction, err := service.PredictSurferCountAdvanced(surferdata.PredictionParams{
			Hour:             hour,
			WaterTemp:        waterTemp,
			AirTemp:          airTemp,
			WeatherCondition: weatherConditionValue,
			WaterLevel:       waterLevel,
		})

		w.Header().Set("Content-Type", "application/json")

		if err != nil {
			// If it's an ML 429/503, set status + Retry-After, but still return the body (degraded).
			if me, ok := err.(*surferdata.MLError); ok && (me.Status == 429 || me.Status == 503) {
				if me.RetryAfterSeconds != nil {
					w.Header().Set("Retry-After", strconv.Itoa(*me.RetryAfterSeconds))
				}
				w.WriteHeader(me.Status)
				_ = json.NewEncoder(w).Encode(prediction)
				return
			}
			// Otherwise 500
			http.Error(w, "Could not compute prediction: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// Return the response as JSON
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(prediction)

	}
}

func handleSurferEntries(service *surferdata.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			entries, err := service.GetAllEntries()
			if err != nil {
				http.Error(w, "Failed to fetch entries", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(entries)

		case http.MethodPost:
			var input struct {
				Count     int       `json:"count"`
				Time      time.Time `json:"timestamp"` // optional
				WaterTemp *float64  `json:"water_temperature,omitempty"`
			}

			if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
				http.Error(w, "Invalid input", http.StatusBadRequest)
				return
			}
			if input.Count < 0 {
				http.Error(w, "Surfer count must be positive", http.StatusBadRequest)
				return
			}

			if err := service.AddEntry(input.Count, input.Time, input.WaterTemp); err != nil {
				log.Printf("Failed to add entry: %v", err)
				http.Error(w, "Failed to save entry", http.StatusInternalServerError)
				return
			}

			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]string{"message": "Entry saved"})

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func HandleWaterHistory(service *conditions.WaterDataService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		history, err := service.GetHistoricalWaterLevels()
		if err != nil {
			http.Error(w, "Failed to fetch historical water levels", http.StatusInternalServerError)
			fmt.Println("❌ Scraper error:", err)
			return
		}
		fmt.Printf("📊 Scraper returned %d entries\n", len(history))
		// for _, h := range history {
		// 	fmt.Println("📅", h.DateTime, "📏", h.Value)
		// }

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(history)
	}
}

func handleMLHealth() http.HandlerFunc {
	flaskURL := os.Getenv("FLASK_API_URL") // e.g. http://127.0.0.1:5001/predict (we'll swap to /health)
	return func(w http.ResponseWriter, r *http.Request) {
		u, _ := url.Parse(flaskURL)
		u.Path = "/health"
		req, _ := http.NewRequestWithContext(r.Context(), http.MethodGet, u.String(), nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]any{"ready": false, "error": err.Error()})
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	}
}
