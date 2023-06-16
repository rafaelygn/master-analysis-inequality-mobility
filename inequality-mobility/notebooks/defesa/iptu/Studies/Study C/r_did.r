library(did)
library(arrow)
library(dplyr)

path <- "//wsl.localhost/Ubuntu-20.04/home/yoshraf/projects/master-analysis-inequality-mobility/inequality-mobility/notebooks/defesa/iptu/Studies/Study C/"

df_did_geral <- read_parquet(paste0(path, "df_did_C.parquet"))
df_did_l <- read_parquet(paste0(path, "df_did_C_lower.parquet"))
df_did_u <- read_parquet(paste0(path, "df_did_C_upper.parquet"))


# Select conver the block to numeric
df_did_geral$sq <- as.numeric(df_did_geral$sq)
df_did_l$sq <- as.numeric(df_did_l$sq)
df_did_u$sq <- as.numeric(df_did_u$sq)

# Select the outcome
y_col <-  'sum Area Residencial - H'
y_col <-  'sum Area Comercial'
y_col <-  'CA'
y_col <-  'QUANTIDADE DE PAVIMENTOS'

out <- att_gt(yname = y_col,
              gname = "cohort",
              idname = "sq",
              tname = "time",
              data = df_did_geral,
              anticipation = 0,
              #control_group = "nevertreated"
              control_group = "notyettreated"
)


# PLOT
#summary(out)
#ggdid(out)
es <- aggte(out, type = "dynamic")
a <- ggdid(es)
a$data <- subset(a$data, between(a$data$year, -5, 15))
a
