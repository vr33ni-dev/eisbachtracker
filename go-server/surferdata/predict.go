package surferdata

import (
	"context"
	"math"

	"github.com/vr33ni/eisbachtracker-pwa/go-server/conditions"
)

type PredictionParams struct {
	Hour             int
	WaterTemp        *float64
	AirTemp          *float64
	WeatherCondition int
	WaterLevel       float64
	WaterFlow        float64
}

// BasePredictionByHour fetches avg surfer count from DB for given hour
func (s *Service) basePredictionByHour(hour int) (float64, error) {
	var avg *float64
	err := s.DB.QueryRow(context.Background(),
		`SELECT AVG(count) FROM surfer_entries WHERE EXTRACT(HOUR FROM timestamp) = $1`,
		hour,
	).Scan(&avg)

	if err != nil {
		return 0, err
	}

	// fallback logic for weird hours (no data or tiny value)
	if avg == nil || *avg < 1 {
		// night hours fallback (basically no one)
		if hour >= 22 || hour <= 5 {
			return 0, nil // super low base
		}
		return 1, nil // minimal base for daytime
	}

	return *avg, nil
}

func (s *Service) PredictSurferCountAdvanced(params PredictionParams) (interface{}, error) {
	base, err := s.basePredictionByHour(params.Hour)
	if err != nil {
		return nil, err
	}

	weatherData := &conditions.WeatherData{
		Temp:      safeFloat(params.AirTemp),
		Condition: params.WeatherCondition,
	}
	factor := calculateFactor(params.Hour, params.WaterTemp, weatherData, params.WaterLevel, params.WaterFlow)
	ruleBased := int(math.Round(base * factor))
	if ruleBased < 0 {
		ruleBased = 0
	}

	mlParams := MLPredictionParams{
		Hour:             params.Hour,
		WaterTemp:        safeFloat(params.WaterTemp),
		AirTemp:          safeFloat(params.AirTemp),
		WaterLevel:       params.WaterLevel,
		WeatherCondition: params.WeatherCondition,
	}
	mlPrediction, explanation, err := s.PredictSurferCountML(mlParams)
	if err != nil {
		// If the ML service is waking up (429/503), return a degraded body AND the error.
		if me, ok := err.(*MLError); ok && (me.Status == 429 || me.Status == 503) {
			resp := map[string]any{
				"hour":              params.Hour,
				"water_temperature": safeFloat(params.WaterTemp),
				"air_temperature":   safeFloat(params.AirTemp),
				"weather_condition": params.WeatherCondition,
				"water_level":       params.WaterLevel,
				"prediction":        ruleBased,
				"explanation":       map[string]float64{},
				"degraded":          true,
				"notice":            "Model is waking up (Render free tier). Showing fallback estimate.",
				"source":            "rule_based_fallback",
			}
			if me.RetryAfterSeconds != nil {
				resp["retry_after_seconds"] = *me.RetryAfterSeconds
			}
			return resp, me // let the HTTP layer set status + headers
		}
		// Other errors bubble up as 500.
		return nil, err
	}

	return map[string]any{
		"hour":              params.Hour,
		"water_temperature": safeFloat(params.WaterTemp),
		"air_temperature":   safeFloat(params.AirTemp),
		"weather_condition": params.WeatherCondition,
		"water_level":       params.WaterLevel,
		"prediction":        mlPrediction,
		"explanation":       explanation,
		"degraded":          false,
		"source":            "ml",
	}, nil
}
