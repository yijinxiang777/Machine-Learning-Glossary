library(DT)
testDF <- data.frame(text=paste0("Section", 1:5), 
                     linksTest1=paste0("<a href=#Section", 1:5, ">Section", 1:5, "</a>"))
datatable(testDF, rownames=FALSE, escape=FALSE)


##Another try

data <- readLines("/Users/xiangyijin/Desktop/Machine-learning/Code_glossary.Rmd")
getwd()
library(stringr)
library(dplyr)
yaml <- which(data == "---")
head <- c(data[yaml[1]:yaml[2]], "\n")
data <- data[(yaml[2]+1):length(data)]

# Indexing lines
# the start palce of each term- Every line start with ### will be the term
def_start <- which(stringr::str_detect(data,  "###"))
# the end place of each term - the start place of the next term -1 
def_end <- c(def_start[2:length(def_start)] - 1, length(data))

def_ranges <- dplyr::data_frame(term = data[def_start],
                                start = def_start,
                                end = def_end) %>%
  # sort the data frame by term
  dplyr::arrange(term) %>%
  dplyr::mutate(new_start = 
                  cumsum(
                    c(1, (end-start+1)[-length(term)])
                  )
  ) %>%
  dplyr::mutate(new_end = new_start + (end-start))

# Create ordered definition list
data2 <- rep(NA, length(data))
for (i in seq_along(def_ranges$term)) {
  start <- def_ranges$start[i]
  end <- def_ranges$end[i]
  n_start <- def_ranges$new_start[i]
  n_end <- def_ranges$new_end[i]
  data2[n_start:n_end] <- data[start:end]
}

# Rewrite rmd

data2 <- c(head, data2[!is.na(data2)])
writeLines(paste(data2, collapse = "\n"))

