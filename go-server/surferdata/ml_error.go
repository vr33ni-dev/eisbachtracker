package surferdata

import "fmt"

type MLError struct {
	Status            int
	Body              string
	RetryAfterSeconds *int
}

func (e *MLError) Error() string { return fmt.Sprintf("ML HTTP %d: %s", e.Status, e.Body) }
