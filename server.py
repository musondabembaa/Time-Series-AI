import grpc
from concurrent import futures
import time
from fastapi import FastAPI
import uvicorn

import weather_pb2
import weather_pb2_grpc

app = FastAPI(title="Onboarders Time Series Prediction Service")

class WeatherService(weather_pb2_grpc.WeatherServiceServicer):
    def CreateForecast(self, request, context):
        # Extract data from the request
        data = request.data
        periods = request.periods
        model_params = request.model_parameters
        return_components = request.return_components
        
        # Dummy forecast data (you can replace this with actual logic)
        forecast_dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        forecast_values = [25.0, 26.0, 27.0]
        forecast_lower_bound = [24.0, 25.0, 26.0]
        forecast_upper_bound = [26.0, 27.0, 28.0]

        # Creating response
        response = weather_pb2.ForecastResponse(
            forecast_dates=forecast_dates,
            forecast_values=forecast_values,
            forecast_lower_bound=forecast_lower_bound,
            forecast_upper_bound=forecast_upper_bound
        )

        return response

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    weather_pb2_grpc.add_WeatherServiceServicer_to_server(WeatherService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

@app.on_event("startup")
async def startup_event():
    import threading
    grpc_thread = threading.Thread(target=serve_grpc, daemon=True)
    grpc_thread.start()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
