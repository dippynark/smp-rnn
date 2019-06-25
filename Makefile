DATA_DIR := data
MAX_EPOCH := 1

MINIO_ACCESS_KEY := minio
MINIO_SECRET_KEY := $(shell cat secrets/minio-secret.key)
MINIO_PID_FILE := .minio.pid

ALPHA_VANTAGE_URL := https://www.alphavantage.co
ALPHA_VANTAGE_API_KEY := $(shell cat secrets/alpha-vantage.key)
ALPHA_VANTAGE_STOCK_COUNT := 10

run: data
	python main.py --stock_count=$(ALPHA_VANTAGE_STOCK_COUNT) --train --input_size=1 --lstm_size=128 --max_epoch=$(MAX_EPOCH) --embed_size=8 --access_key $(MINIO_ACCESS_KEY) --secret_key $(MINIO_SECRET_KEY)

#.PHONY: data
data:
	mkdir -p $(DATA_DIR)
	curl -o $(DATA_DIR)/companylist.csv https://www.nasdaq.com/screening/companies-by-industry.aspx\?industry\=Technology\&render\=download
	# https://www.alphavantage.co/documentation/
	# API call frequency is 5 calls per minute and 500 calls per day
	COUNT=0
	for SYMBOL in $(shell cat data/companylist.csv | tail -n +2 | cut -d ',' -f 1 | tr -d '"' | head -n $(ALPHA_VANTAGE_STOCK_COUNT)); do \
		curl -o $(DATA_DIR)/$$SYMBOL.csv "$(ALPHA_VANTAGE_URL)/query?function=TIME_SERIES_DAILY&symbol=$$SYMBOL&apikey=$(ALPHA_VANTAGE_API_KEY)&datatype=csv&outputsize=full"; \
		((COUNT = COUNT + 1)); \
		if [ `echo "$$COUNT % 5" | bc` -eq 0 ]; then \
		  sleep 60; \
		fi; \
	done