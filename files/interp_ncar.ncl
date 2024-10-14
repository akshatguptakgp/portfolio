; Script to interpolate from Hybrid to Pure Pressure coordinates
; Created By : Aman Gupta @ag4680
; Reading in hybrid data from file
; Interpolates multiple files
; Can add features to compute zonal mean statistics and Eddy statistics too
; The script taked time to start because the netCDF files are being created in the process.

; to add more variables, just define them in the vardef and main interpolation
; part

        t_start = 8761
        t_end   = 9855
        delt    = 365
        N = (t_end - t_start + 1)/delt
        print("Preparing to interpolate " + tostring(N) + " files")
        inttype = 2        ; 1 - Linear, 2 - Log, 3 - LogLog

        ;idir = "/scratch/ag4680/model_output/CESM/archive/f09l40k2h3_aoa/atm/hist/"
        idir = "/scratch/ag4680/model_output/CESM/archive/ne30np4l40k2h3_aoa/atm/hist/"
        ifile = addfile(idir + "atmos_1day_d0" + tostring(t_start+delt-1) + ".nc" , "r")  ; a NetCDF file
        ;print(ifile)

        ; Good for L40 runs
        pnew = (/ .01, .03, .07, .15, .27, .48, .78, 1.23, 1.87, 2.74, 3.91, 5.46, 7.46, 10., 13., 17., 22., \
                28., 35., 44., 50., 65., 80., 100., 115., 135., 160., 190., 220., 250., \
                300., 350., 400., 450., 500., 575., 650., 750., 850., 950. /) ; desired output levels in hPa
                                                                      ; both top-to-bottom and bottom-to-top works

        ;pnew = (/ 0.01, 0.02, 0.03, 0.04, 0.06, 0.09, 0.12, 0.17, 0.25, 0.30, 0.40, 0.55, 0.70, 0.85, 1.10, 1.40, \
        ;         1.75, 2.00, 2.50, 3.00, 3.50, 4.25, 5.00, 5.90, 6.90, 8., 9.30, 10., 12.50, 14., 16.,   \
        ;        18., 20., 23., 26., 30., 33., 37., 41., 45., 50., 56., 62., 68., 75., 83., 91., 100., 110.,\
        ;       120., 130., 140., 155., 165., 180., 195., 210., 230., 250., 270., 285., 310., 330., 350., \
        ;        380., 400., 440., 460., 500., 530., 570., 600., 640., 680., 725., 770., 820., 850., 925., 975. /)

        ; Add alternative for L80 runs

        pnew@units = "hPa"                  ; Setting the units attribute for this variable
        pnew!0 = "lev"                      ; This is important - Tnew inherits this name

; Loading dimension vectors
        print("Loading dimension vectors...")
        hyam = ifile->hyam                   ; Should be in [0,1]. Divide by 10^5 if reqd
        ;hyam = hyam_temp/100000.                 ; Not needed for NCAR models
        hybm = ifile->hybm
        time = ifile->time
        lev  = ifile->lev
        lat  = ifile->lat
        ;lat!0 = "lat"
        lon  = ifile->lon
        ;lon!0 = "lon"
; Needed 3D variables
        ps   = ifile->PS
; Other constants
        P0mb = 1000.

; hybm and hyam should be at full model levels(NLEV), not half(NLEV+1)
; Not needed for NCAR models since hyam, hybm are full model level coeffs
        ;nhalf  = dimsizes(hybm_temp)
        ;hyam = (hyam_temp(1:nhalf-1) + hyam_temp(0:nhalf-2))/2
        ;hybm = (hybm_temp(1:nhalf-1) + hybm_temp(0:nhalf-2))/2
i=1
do while(i.le.N)

;set ifile name
        print("Interpolating days " + tostring(t_start + (i-1)*delt) + " to " + tostring(t_start -1 + (i)*delt)  + "...")
        ifile = addfile (idir + "atmos_1day_d0" + tostring(t_start-1+ (i*delt)) + ".nc" , "r")
        time = ifile->time
        ; Needed 3D variables
        ps   = ifile->PS


; Writing to file
; Following the Efficient way to write to the output file
; Much faster and more accurate, systematic and descriptive
; Copied from the NCL website

        ntim  = dimsizes(time)                 ; get dimension sizes
        nlev  = dimsizes(pnew)
        nlat  = dimsizes(lat)
        nlon  = dimsizes(lon)

        diro = idir                            ; Output Directory
        filo = "plev_atmos_1day_d0" + tostring(t_start-1+ (i*delt)) + ".nc"  ; Output file
        system("/bin/rm -f " + diro + filo)    ; remove if exists
        fout  = addfile(diro + filo, "c")  ; open output file

;===================================================================
; explicitly declare file definition mode. Improve efficiency.
;===================================================================
        setfileoption(fout,"DefineMode",True)

;===================================================================
;===================================================================
; create global attributes of the file
;===================================================================
;       fAtt               = True            ; assign file attributes
;       fAtt@title         = "NCL Efficient Approach to netCDF Creation"
;       fAtt@source_file   =  "atmos_1day_d06000.nc"
;       fAtt@Conventions   = "None"
;       fAtt@creation_date = systemfunc ("date")
;       fileattdef( fout, fAtt )            ; copy file attributes

;===================================================================
; predefine the coordinate variables and their dimensionality
; Note: to get an UNLIMITED record dimension, we set the dimensionality
; to -1 (or the actual size) and set the dimension name to True.
;===================================================================
        dimNames = (/"time", "lev", "lat", "lon"/)
        dimSizes = (/ -1   ,  nlev, nlat,  nlon/)
        dimUnlim = (/ True , False, False, False/)
        filedimdef(fout,dimNames,dimSizes,dimUnlim)

;===================================================================
; predefine the the dimensionality of the variables to be written out
;===================================================================
; Here we are using NCL functions to facilitate defining
; each variable's dimension name(s) and type.
; The following could be replaced with explicit, user defined dimension
; names different from those associated with the variable in memory.
; Say, PS(time,lat,lon) in the NCL script. They could be redefined for the file via:
; filevardef(fout, "PS"   ,typeof(PS) ,(/"TIME","latitude","longitude"/))
;===================================================================
       filevardef(fout, "time" ,typeof(time),getvardims(time))
       filevardef(fout, "lev"  ,typeof(pnew),getvardims(pnew) )
       filevardef(fout, "lat"  ,typeof(lat),getvardims(lat))
       filevardef(fout, "temp" ,"float",(/"time", "lev", "lat", "lon"/))
       filevardef(fout, "ucomp" ,"float",(/"time", "lev", "lat", "lon"/))
       filevardef(fout, "vcomp" ,"float",(/"time", "lev", "lat", "lon"/))

;===================================================================
; Copy attributes associated with each variable to the file
; All attributes associated with each variable will be copied.
;====================================================================
       filevarattdef(fout,"time" ,time)                    ; copy time attributes
       filevarattdef(fout,"lev"  ,pnew)                    ; copy lev attributes
       filevarattdef(fout,"lat"  ,lat)                     ; copy lat attributes
       filevarattdef(fout,"lon"  ,lon)                     ; copy lon attributes

;===================================================================
; explicitly exit file definition mode. **NOT REQUIRED**
;===================================================================
        setfileoption(fout,"DefineMode",False)

;===================================================================
; output only the data values since the dimensionality and such have
; been predefined. The "(/", "/)" syntax tells NCL to only output the
; data values to the predefined locations on the file.
;====================================================================
       fout->time   = (/time/)
       fout->lev    = (/pnew/)
       fout->lat    = (/lat/)
       fout->lon    = (/lon/)
;===========================================================================
; Declarations done; Now Take each 4D var one by one, interpolate and write.
;===========================================================================

                j=0
                do while (j .lt. delt)
                        print("File "+i+"/"+N+", Day "+(j+1)+"/"+delt)
                        ps_now = ps(j,:,:)

                        ; Interpolating temperature
                        T    = ifile->T(j,:,:,:)        ; Loading Variables
                        Tnew = vinth2p (T(lev|:,lat|:,lon|:),hyam,hybm,pnew,ps_now(lat|:,lon|:),inttype,P0mb,1,True)
                        fout->temp(j,:,:,:)   = (/Tnew/)

                        ; Interpolating Zonal Winds
                        U    = ifile->U(j,:,:,:)
                        Unew = vinth2p (U(lev|:,lat|:,lon|:),hyam,hybm,pnew,ps_now(lat|:,lon|:),inttype,P0mb,1,True)
                        fout->ucomp(j,:,:,:)   = (/Unew/)

                        ; Interpolating Meridional Winds
                        V    = ifile->V(j,:,:,:)
                        Vnew = vinth2p (V(lev|:,lat|:,lon|:),hyam,hybm,pnew,ps_now(lat|:,lon|:),inttype,P0mb,1, True)
                        fout->vcomp(j,:,:,:)   = (/Vnew/)
                j = j+1
                end do

        i = i+1
end do



