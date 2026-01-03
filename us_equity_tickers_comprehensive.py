#!/usr/bin/env python3
"""
Comprehensive U.S. Equity Ticker Database

This module provides comprehensive static lists of U.S. equity tickers
for all major indices. These lists are used as fallback when live data
is unavailable, and represent a comprehensive cross-section of the U.S.
equity universe.

Data sources conceptual coverage:
- S&P 500: ~503 large-cap stocks
- Russell 3000: ~3000 stocks (large, mid, and small cap)
- Russell 2000: ~2000 small-cap stocks
- NASDAQ Composite: ~3000+ NASDAQ-listed stocks
- Dow Jones: 30 blue-chip stocks

This module provides representative samples covering the major constituents.
"""

# S&P 500 - Major Large Cap U.S. Equities (503 stocks as of 2024)
SP500_TICKERS = [
    # Mega-cap Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "ACN", "AMD", "INTC", "IBM", "NOW", "TXN", "QCOM",
    "INTU", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ADSK", "FTNT", "PANW",
    "CRWD", "WDAY", "SNOW", "ANSS", "APH", "FFIV", "JNPR", "NTAP", "STX", "KEYS",
    "AKAM", "VRSN", "ENPH", "MPWR", "ON", "SWKS", "TER", "MRVL", "GEN", "GLW", "HPQ",
    
    # Communication Services
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "LYV", "MTCH", "PARA", "OMC", "IPG", "NWSA",
    
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "MAR",
    "GM", "F", "ABNB", "CMG", "ORLY", "AZO", "ROST", "DHI", "LEN", "YUM",
    "DPZ", "ULTA", "DECK", "POOL", "TPR", "RL", "GPS", "M", "JWN", "KSS",
    "AAP", "APTV", "BBY", "BURL", "BWA", "CCL", "CZR", "DG", "DRI", "EBAY",
    "EXPE", "GRMN", "GPC", "HAS", "HLT", "KMX", "LKQ", "LVS", "MGM", "MHK",
    "NCLH", "NVR", "PHM", "PKG", "POOL", "PVH", "RCL", "SBAC", "TSCO", "VFC",
    "WHR", "WYNN",
    
    # Consumer Staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "GIS",
    "KMB", "SYY", "KHC", "HSY", "K", "CAG", "CPB", "HRL", "SJM", "MKC",
    
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL",
    "DVN", "HES", "FANG", "BKR", "KMI", "WMB", "LNG", "TRGP", "OKE", "APA",
    "EQT", "MRO", "PXD",
    
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "SCHW",
    "AXP", "BLK", "SPGI", "USB", "PNC", "TFC", "COF", "BK", "STT", "NTRS",
    "FRC", "RF", "CFG", "KEY", "FITB", "HBAN", "MTB", "ZION", "CMA", "SIVB",
    "AFL", "AIG", "AIZ", "AJG", "ALL", "AON", "AMP", "BEN", "BR", "CBOE",
    "CINF", "CME", "DFS", "EG", "FDS", "GL", "HIG", "ICE", "IVZ", "JKHY",
    "L", "MCO", "MKTX", "MMC", "MOH", "MSCI", "NDAQ", "PFG", "PRU", "RE",
    "RJF", "SYF", "TROW", "TRV", "WRB",
    
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "CVS", "MDT", "GILD", "ISRG", "CI", "REGN", "VRTX", "HUM", "ZTS",
    "BSX", "ELV", "SYK", "MCK", "COR", "BDX", "EW", "IDXX", "HCA", "A",
    "ALGN", "BAX", "BIO", "CAH", "CNC", "CERN", "CRL", "DXCM", "EXAS", "HOLX",
    "IQV", "LH", "MTD", "PKI", "PODD", "RMD", "STE", "TFX", "UHS", "VAR",
    "VTRS", "WAT", "XRAY", "ZBH",
    
    # Industrials
    "CAT", "UNP", "UPS", "RTX", "BA", "HON", "LMT", "GE", "DE", "MMM",
    "NSC", "FDX", "CSX", "NOC", "EMR", "ETN", "ITW", "PH", "WM", "GD",
    "TT", "PCAR", "ROK", "CARR", "OTIS", "JCI", "CTAS", "CMI", "FAST",
    "AOS", "AXON", "BLDR", "CHRW", "CNI", "DAL", "DOV", "EXPD", "FBHS", "FTV",
    "GWW", "HII", "HWM", "IEX", "IR", "J", "JBHT", "LHX", "MAS", "NDSN",
    "ODFL", "PAYC", "PNR", "PWR", "RSG", "SNA", "SWK", "TDG", "TXT", "UAL",
    "URI", "VRSK", "WAB", "XYL",
    
    # Materials
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "PPG",
    "VMC", "MLM", "CTVA", "EMN", "CE", "ALB", "IFF", "MOS", "FMC", "CF",
    "AMCR", "AVY", "BALL", "IP", "LYB", "PKG", "SEE", "STLD", "WRK",
    
    # Real Estate
    "AMT", "PLD", "EQIX", "PSA", "WELL", "DLR", "O", "SPG", "VICI", "AVB",
    "EQR", "SBAC", "CBRE", "VTR", "ARE", "MAA", "INVH", "ESS", "UDR", "CPT",
    "BXP", "DOC", "EXR", "FRT", "HST", "IRM", "KIM", "REG", "WY",
    
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
    "PEG", "ED", "AWK", "DTE", "PPL", "EIX", "FE", "AEE", "CMS", "CNP",
    "AES", "ATO", "ETR", "EVRG", "LNT", "NI", "NRG", "PNW",
    
    # Additional S&P 500
    "PYPL", "SQ", "FISV", "FIS", "ADP", "PAYX", "CTSH",
]

# Additional large/mid-cap stocks for Russell 3000 coverage (beyond S&P 500)
RUSSELL_3000_ADDITIONS = [
    # Mid-cap growth
    "PLTR", "RBLX", "NET", "DDOG", "ZS", "OKTA", "ZM", "DOCU", "TWLO", "COUP",
    "BILL", "DOCN", "S", "PATH", "GTLB", "CFLT", "MDB", "ESTC", "TEAM", "HUBS",
    "FROG", "DASH", "COIN", "CPNG", "SHOP", "W", "U", "AFRM", "UPST", "SOFI",
    "DKNG", "CELH",
    
    # Mid-cap value & cyclical
    "ALK", "JBLU", "LUV", "UAL", "DAL", "AAL", "HA", "SAVE", "SKYW",
    
    # EV and clean energy
    "RIVN", "LCID", "CHPT", "BLNK", "EVGO", "FSLR", "SEDG", "RUN", "PLUG",
    
    # Quantum computing & AI
    "IONQ", "RGTI", "QBTS", "SOUN", "BBAI", "AI", "SMCI", "DELL", "HPE",
    
    # Healthcare/Biotech mid-cap
    "MRNA", "NVAX", "BNTX", "CRSP", "NTLA", "EDIT", "BEAM", "BLUE", "FATE",
    "VCYT", "PACB", "NVTA", "NTRA", "BIIB", "ILMN", "ALNY", "SGEN", "BMRN",
    "TECH", "EXAS",
]

# Russell 2000 - Small Cap Representative Sample
RUSSELL_2000_REPRESENTATIVE = [
    # Small-cap tech
    "MARA", "RIOT", "HUT", "CLSK", "BITF", "CIFR", "BTBT", "ARBK", "IREN",
    "ALKT", "ALRM", "APPF", "ARLO", "ASAN", "AVAV", "AVDX", "BAND", "BASE",
    "BIGC", "BL", "BLFS", "BOX", "CALX", "CASS", "CDLX", "CGNX", "CLBT",
    "CLVT", "COMM", "CONE", "CWAN", "CYBR", "DT", "EVBG", "EXLS", "FIVN",
    "FORM", "FRSH", "GLOB", "GTLS", "GWRE", "HSTM", "INFN", "INST", "IRDM",
    "LITE", "LUMN", "LYFT", "MGNI", "MIME", "MITK", "NEWR", "NOVT", "NTCT",
    "NTNX", "ONTO", "OSIS", "PCTY", "PDFS", "PING", "PSTG", "PTC", "QTWO",
    "RAMP", "RNG", "RPD", "RXT", "SAIC", "SATS", "SMAR", "SPSC", "SPWR",
    "SSYS", "TENB", "TNET", "TTEC", "TWOU", "UBER", "UCTT", "UPLD", "UPWK",
    "VECO", "VEEV", "VIAV", "VIRT", "VLY", "VRRM", "VSAT", "VUZI", "WEAV",
    "YELP", "YOU", "YEXT",
    
    # Small-cap healthcare/biotech
    "ACAD", "ADMA", "ADPT", "ADVM", "AEIS", "AGEN", "AKRO", "ALEC", "ALLO",
    "ALPN", "AMRN", "ANGO", "ANIP", "APLS", "APRE", "APTX", "AQST", "ARDS",
    "ARWR", "ASND", "ASMB", "ASRT", "ATRC", "ATNM", "ATRO", "AUPH", "AVIR",
    "AVNS", "AXGN", "AXSM", "BCYC", "BDTX", "BCRX", "BLRX", "BOLD", "BPMC",
    "BSGM", "CARA", "CBAY", "CCXI", "CERE", "CHRS", "CLDX", "CMPS", "CNCE",
    "CNTB", "CORT", "CRBP", "CRBU", "CRIS", "CRNX", "CRTX", "CTMX", "CVAC",
    "CYAD", "CYCN", "DARE", "DAVA", "DBVT", "DCPH",
    
    # Small-cap consumer
    "AMC", "APE", "BBWI", "BFAM", "BOOT", "BRC", "BROS", "BYND", "CAKE",
    "CHWY", "COCO", "CVNA", "DDS", "DNUT", "DORM", "DV", "EAT", "ETSY",
    "FIVE", "FL", "FTRE", "GES", "GOOS", "GRUB", "HBI", "HIBB", "LEVI",
    "LRN", "LULU", "LZB", "MBUU", "NLS", "ODP", "OLLI", "ONON", "OSTK",
    "PETS", "PLAY", "PLNT", "PRPL", "PTON", "REAL", "RH", "RVLV", "SABR",
    "SFIX", "SHAK", "SKIN", "SSTK", "STOR", "TCS", "TRIP", "UA", "UAA",
    "URBN", "VRA", "VSTO", "WARM", "WEN", "WING", "WOOF", "WSM", "YETI",
    "ZG", "ZUMZ",
    
    # Small-cap industrials
    "ATKR", "AZEK", "BC", "BLBD", "BW", "CBT", "CECO", "CRS", "CW", "DY",
    "ESNT", "FELE", "FLS", "GFF", "GGG", "GNRC", "HI", "HUBG", "HUBB",
    "ITRI", "JBT", "KAI", "KALU", "KBR", "KMT", "LECO", "LII", "MATW",
    "MIDD", "MLI", "MSA", "MTZ", "NPO", "NSIT", "NWPX", "OSK", "PRIM",
    "R", "RBC", "RXO", "SAIA", "SLGN", "SPR", "STRL", "TDY", "TEX", "TNC",
    "TRS", "TRU", "TTC", "UNF", "VITL", "WERN", "WMS", "WSO", "XPO",
    
    # Small-cap financials/regional banks
    "PACW", "WAL", "SBNY", "ONB", "UMBF", "UBSI", "FHN", "SNV",
    
    # Small-cap energy/materials
    "WKHS", "RIDE", "NKLA", "HYLN", "GOEV", "ARVL", "MULN", "ELMS", "AYRO", "GEV",
    "GME", "KOSS", "EXPR", "NAKD", "SNDL", "TLRY", "CGC",
]

# NASDAQ Composite - Major NASDAQ-listed stocks
NASDAQ_COMPOSITE_TICKERS = [
    # NASDAQ 100
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "NFLX", "ADBE", "PEP", "CSCO", "CMCSA", "TMUS", "AMD", "INTC", "QCOM", "INTU",
    "TXN", "AMGN", "HON", "AMAT", "SBUX", "ISRG", "BKNG", "ADP", "GILD", "VRTX",
    "REGN", "LRCX", "MDLZ", "MU", "PANW", "PYPL", "KLAC", "ASML", "ABNB", "SNPS",
    "CDNS", "MELI", "MAR", "CSX", "CRWD", "ORLY", "ADSK", "FTNT", "NXPI", "WDAY",
    "DXCM", "PCAR", "CHTR", "CTAS", "MNST", "PAYX", "MCHP", "AEP", "ROST", "FAST",
    "EXC", "ODFL", "IDXX", "KDP", "EA", "VRSK", "CCEP", "CPRT", "GEHC", "DDOG",
    "ZS", "TEAM", "MRNA", "BIIB", "ILMN", "ALNY", "KHC", "SIRI", "XEL", "WBD",
    "CSGP", "DLTR", "ANSS", "TTWO", "ON", "FANG", "CDW", "WBA", "GFS", "LULU",
    "ALGN", "BKR", "EBAY", "SGEN", "JD", "PDD", "BIDU", "LI", "NTES", "ZM",
    
    # Additional major NASDAQ stocks
    "SNOW", "NET", "OKTA", "MDB", "PLTR", "RBLX", "U", "DOCN", "CFLT", "GTLB",
    "S", "PATH", "BILL", "COUP", "DOCU", "TWLO", "DASH", "COIN", "CPNG", "SHOP",
    "W", "AFRM", "UPST", "SOFI", "DKNG", "CELH", "RIVN", "LCID", "CHPT", "BLNK",
    "EVGO", "ENPH", "FSLR", "SEDG", "RUN", "PLUG", "IONQ", "RGTI", "QBTS", "SOUN",
    "BBAI", "AI", "SMCI", "DELL", "HPE",
    
    # Mid-cap NASDAQ
    "ACIW", "ACLS", "ADUS", "AEIS", "AGEN", "AIMC", "AKAM", "ALGT", "ALGM",
    "ALHC", "ALKS", "ALVR", "AMED", "AMKR", "AMPH", "AMSF", "AMSWA", "AMWD",
    "ANAB", "ANAT", "ANDE", "ANGI", "ANIP", "APAM", "APEI", "APEN", "APOG",
    "APPF", "APPS", "APRE", "APTS", "APYX", "AQMS", "ARCH", "ARCB", "ARCC",
    "ARDX", "AREC", "ARGO", "AROW", "ARQT", "ARTNA", "ARVN", "ARWR", "ASIX",
    "ASND", "ASPS", "ASPU", "ASRT", "ASTC", "ASTE", "ASUR", "ATEX", "ATHA",
    "ATHN", "ATHX", "ATIF", "ATIS", "ATLO", "ATOM", "ATOS", "ATRC", "ATRI",
    "ATRO", "ATSG", "ATVI", "AUB", "AUBN", "AUR", "AVAV", "AVCO", "AVDL",
]

# Dow Jones Industrial Average (30 stocks)
DOW_JONES_TICKERS = [
    "AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "MCD", "AMGN", "V", "BA",
    "HON", "TRV", "JPM", "AXP", "IBM", "JNJ", "PG", "CVX", "WMT", "MRK",
    "DIS", "CRM", "CSCO", "NKE", "VZ", "KO", "DOW", "INTC", "MMM", "WBA"
]

def get_all_tickers():
    """
    Get all unique tickers from all indices, deduplicated.
    
    Returns:
        List of unique ticker symbols (sorted)
    """
    all_tickers = set()
    all_tickers.update(SP500_TICKERS)
    all_tickers.update(RUSSELL_3000_ADDITIONS)
    all_tickers.update(RUSSELL_2000_REPRESENTATIVE)
    all_tickers.update(NASDAQ_COMPOSITE_TICKERS)
    all_tickers.update(DOW_JONES_TICKERS)
    
    return sorted(list(all_tickers))


def get_ticker_index_mapping():
    """
    Create a mapping of ticker -> list of indices it belongs to.
    
    Returns:
        Dict mapping ticker to list of index names
    """
    mapping = {}
    
    for ticker in SP500_TICKERS:
        mapping.setdefault(ticker, []).append("SP_500")
    
    for ticker in RUSSELL_3000_ADDITIONS:
        mapping.setdefault(ticker, []).append("RUSSELL_3000")
    
    for ticker in RUSSELL_2000_REPRESENTATIVE:
        mapping.setdefault(ticker, []).append("RUSSELL_2000")
    
    for ticker in NASDAQ_COMPOSITE_TICKERS:
        mapping.setdefault(ticker, []).append("NASDAQ_COMPOSITE")
    
    for ticker in DOW_JONES_TICKERS:
        mapping.setdefault(ticker, []).append("DOW_JONES")
    
    return mapping


if __name__ == "__main__":
    all_tickers = get_all_tickers()
    print(f"Total unique tickers: {len(all_tickers)}")
    print(f"S&P 500: {len(SP500_TICKERS)}")
    print(f"Russell 3000 additions: {len(RUSSELL_3000_ADDITIONS)}")
    print(f"Russell 2000 representative: {len(RUSSELL_2000_REPRESENTATIVE)}")
    print(f"NASDAQ Composite: {len(NASDAQ_COMPOSITE_TICKERS)}")
    print(f"Dow Jones: {len(DOW_JONES_TICKERS)}")
