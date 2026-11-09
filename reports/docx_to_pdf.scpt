-- Export a .docx to PDF using the locally installed Microsoft Word.
-- Usage: osascript reports/docx_to_pdf.scpt <absolute .docx path> <absolute .pdf path>
-- Word has no command-line exporter, so this drives the app. If Word is not installed, not
-- licensed, or is showing a dialog, export manually with File > Save as > PDF.
on run argv
	set inPath to item 1 of argv
	set outPath to item 2 of argv
	tell application "Microsoft Word"
		set wasRunning to running
		with timeout of 900 seconds
			activate
			open POSIX file inPath
			set theDoc to active document
			save as theDoc file name outPath file format format PDF
			close theDoc saving no
			if not wasRunning then quit
		end timeout
	end tell
	return "exported " & outPath
end run
