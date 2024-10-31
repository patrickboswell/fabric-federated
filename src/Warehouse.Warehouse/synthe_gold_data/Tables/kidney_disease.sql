CREATE TABLE [synthe_gold_data].[kidney_disease] (

	[ENCOUNTER_ID] varchar(8000) NULL, 
	[ED_FLAG] bit NULL, 
	[LOS] int NULL, 
	[START] datetime2(6) NULL, 
	[STOP] datetime2(6) NULL, 
	[TOTAL_CLAIM_COST] varchar(8000) NULL, 
	[PATIENT_ID] varchar(8000) NULL, 
	[AGE] decimal(23,0) NULL, 
	[INCOME_CAT] varchar(8000) NULL, 
	[FATAL_ENCOUNTER] bit NULL, 
	[DIABETES_FLAG] bit NULL, 
	[GFR_MIN] real NULL, 
	[GFR_MAX] real NULL, 
	[AL_CR_MIN] real NULL, 
	[AL_CR_MAX] real NULL, 
	[ORGANIZATION_NAME] varchar(8000) NULL, 
	[PROVIDER_NAME] varchar(8000) NULL, 
	[PROVIDER_SPECIALITY] varchar(8000) NULL, 
	[PAYER_NAME] varchar(8000) NULL
);

