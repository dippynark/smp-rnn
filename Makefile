DATA_DIR := data

MINIO_ACCESS_KEY := minio
MINIO_SECRET_KEY := $(shell cat secrets/minio-secret.key)

ALPHA_VANTAGE_URL := https://www.alphavantage.co
ALPHA_VANTAGE_API_KEY := $(shell cat secrets/alpha-vantage.key)
ALPHA_VANTAGE_STOCK_COUNT := 500

run:
	python main.py --access_key $(MINIO_ACCESS_KEY) --secret_key $(MINIO_SECRET_KEY)

.PHONY: data
data:
	mkdir -p $(DATA_DIR)
	curl -o $(DATA_DIR)/companylist.csv https://www.nasdaq.com/screening/companies-by-industry.aspx\?industry\=Technology\&render\=download
	# https://www.alphavantage.co/documentation/
	# API call frequency is 5 calls per minute and 500 calls per day
	DOWNLOAD_COUNT=0; \
	for SYMBOL in $(shell cat data/companylist.csv | sed 's/          //g' | tail -n +2 | cut -d ',' -f 1 | tr -d '"' | head -n $(ALPHA_VANTAGE_STOCK_COUNT)); do \
		if [ -f $(DATA_DIR)/$$SYMBOL.csv ]; then \
			continue; \
		fi; \
		if [ `echo "$$DOWNLOAD_COUNT % 5" | bc` -eq 0 ]; then \
			echo "Sleeping..."; \
			sleep 60; \
		fi; \
		curl -o $(DATA_DIR)/$$SYMBOL.csv "$(ALPHA_VANTAGE_URL)/query?function=TIME_SERIES_DAILY&symbol=$$SYMBOL&apikey=$(ALPHA_VANTAGE_API_KEY)&datatype=csv&outputsize=full"; \
		((DOWNLOAD_COUNT=DOWNLOAD_COUNT+1)); \
	done